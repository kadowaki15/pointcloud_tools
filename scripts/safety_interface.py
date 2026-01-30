#!/usr/bin/env python3
"""
safety_interface (ROS2 Humble)

前提（あなたの今の環境）:
- /tf は publisher 0（動的TFなし）
- /tf_static は RealSense が出している（静的TFのみ）
- /tracked_markers は frame_id = camera_depth_optical_frame（optical座標）

この版は optical座標の軸（前方=z, 左右=x）に合わせて距離判定を行い、
「危険でSTOPしたら最低 stop_min_hold_sec は停止を維持」する。

追加（キーボードE-STOP）:
- SPACE: 停止をラッチ（/cmd_vel_safe を必ず 0 にする）
- r    : ラッチ解除
※このノードを起動している端末がアクティブな時だけ効く

追加（急接近 STOP）:
- dmin（前方最短距離）の減少速度 v_close=(d_old-d_new)/dt が閾値を超えたら STOP
- FAR_CROSS_SUPPRESS_APPROACH 中でも「急接近」だけは STOP させる（距離0.45m固定では止めない）
- 急接近STOPしたら最低1秒停止（lock優先：FAR_CROSSがlockを上書きして前進復帰しないように修正）
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Twist, PoseStamped
import time
import math
from collections import deque

# Keyboard E-STOP
import sys
import termios
import tty
import select

try:
    import tf2_ros
    import tf2_geometry_msgs
    TF2_AVAILABLE = True
except Exception:
    TF2_AVAILABLE = False


class SafetyInterface(Node):
    def __init__(self):
        super().__init__('safety_interface')

        # ---------------- topics ----------------
        self.declare_parameter('static_topic', '/static_obstacles')
        self.declare_parameter('tracked_topic', '/tracked_markers')
        self.declare_parameter('intent_topic', '/object_intents')
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('publish_topic', '/cmd_vel_safe')

        # ---------------- loop ----------------
        self.declare_parameter('tick_dt', 0.2)

        # ---------------- speeds ----------------
        self.declare_parameter('default_speed', 0.20)
        self.declare_parameter('slow_speed', 0.10)

        # ---------------- MIN STOP HOLD ----------------
        self.declare_parameter('stop_min_hold_sec', 1.0)

        # ---------------- EMERGENCY STOP ----------------
        self.declare_parameter('emergency_stop_dist', 0.30)
        self.declare_parameter('emergency_hold_sec', 1.5)

        # Emergency gate (front corridor)
        # optical前提: 前方=z、左右=x として扱う
        self.declare_parameter('emg_gate_x_max', 0.80)   # = z_max
        self.declare_parameter('emg_gate_y_half', 0.45)  # = x_half

        # ---------------- APPROACH (dynamic) ----------------
        self.declare_parameter('stop_dist', 0.45)
        self.declare_parameter('slow_dist', 1.00)

        self.declare_parameter('stop_hold_sec', 1.5)
        self.declare_parameter('slow_hold_sec', 1.2)

        # approach gate（optical前提: z_min/z_max として解釈）
        self.declare_parameter('dyn_gate_x_min', 0.10)  # = z_min
        self.declare_parameter('dyn_gate_x_max', 2.00)  # = z_max
        self.declare_parameter('dyn_gate_y_half', 0.45) # = x_half

        # ---------------- CROSSING ----------------
        self.declare_parameter('cross_stop_dist', 0.35)
        self.declare_parameter('cross_slow_dist', 0.80)
        self.declare_parameter('cross_hold_sec', 0.6)
        self.declare_parameter('cross_cache_sec', 0.8)
        self.declare_parameter('far_cross_suppress_sec', 0.8)

        # ---------------- output smoothing ----------------
        self.declare_parameter('use_slew_limiter', True)
        self.declare_parameter('max_accel', 0.15)  # m/s^2
        self.declare_parameter('max_decel', 0.30)  # m/s^2

        # ---------------- STATIC (avoid + hold) ----------------
        self.declare_parameter('front_trigger_dist', 0.70)
        self.declare_parameter('lateral_scan_dist', 1.8)
        self.declare_parameter('path_half_width', 0.35)
        self.declare_parameter('avoidance_angular', 0.6)

        self.declare_parameter('static_cache_sec', 1.2)
        self.declare_parameter('static_on_frames', 3)
        self.declare_parameter('static_off_frames', 6)
        self.declare_parameter('static_avoid_hold_sec', 1.0)

        self.declare_parameter('front_off_dist', 0.85)
        self.declare_parameter('path_off_half_width', 0.50)

        # ---------------- frames ----------------
        self.declare_parameter('base_frame', 'camera_link')

        # ---------------- Keyboard E-STOP ----------------
        self.declare_parameter('kb_estop_enabled', True)

        # ---------------- RAPID APPROACH (rate-based stop) ----------------
        self.declare_parameter('rapid_close_enabled', True)
        self.declare_parameter('rapid_close_v_thresh', 1.0)     # [m/s]
        self.declare_parameter('rapid_close_dist_max', 1.6)     # [m]
        self.declare_parameter('rapid_close_window_sec', 0.5)   # [s]
        self.declare_parameter('rapid_close_min_samples', 3)
        self.declare_parameter('rapid_close_hold_sec', 1.0)     # ★急接近STOPの保持を1秒に

        # load parameters
        for p in self._parameters:
            setattr(self, p, self.get_parameter(p).value)
        self.tick_dt = float(self.get_parameter('tick_dt').value)

        # TF
        if TF2_AVAILABLE:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        else:
            self.tf_buffer = None

        # subs/pubs
        self.sub_static  = self.create_subscription(MarkerArray, self.static_topic,  self.cb_static,  10)
        self.sub_tracked = self.create_subscription(MarkerArray, self.tracked_topic, self.cb_tracked, 10)
        self.sub_intent  = self.create_subscription(String,      self.intent_topic,  self.cb_intent,  10)
        self.sub_cmd     = self.create_subscription(Twist,       self.cmd_topic,     self.cb_cmd,     10)

        self.pub_cmd   = self.create_publisher(Twist,  self.publish_topic, 10)
        self.pub_state = self.create_publisher(String, '/safety_state',    10)

        # internal state
        self.latest_cmd = Twist()
        self.recent_tracked = deque(maxlen=10)

        # rapid-approach history: (t, dmin)
        self._dmin_hist = deque(maxlen=30)

        # static
        self.static_objs = []
        self._last_static_time = 0.0
        self._static_latched = False
        self._static_true_cnt = 0
        self._static_false_cnt = 0
        self._static_avoid_until = 0.0

        # intent cache for crossing
        self._last_cross_time = 0.0
        self._last_cross_rng = float('nan')
        self._far_cross_until = 0.0

        # dynamic hold
        self._dyn_mode = "NORMAL"      # NORMAL / SLOW / STOP
        self._dyn_lock_until = 0.0
        self._dyn_reason = "NORMAL"
        # ★追加: “比較用” reason_key（距離など数値で変動しないキー）
        self._dyn_reason_key = "NORMAL"

        # slew output
        self._v_out = 0.0

        # ---------------- Keyboard E-STOP state ----------------
        self._kb_estop_latched = False
        self._kb_old_term = None

        if bool(self.kb_estop_enabled):
            try:
                fd = sys.stdin.fileno()
                self._kb_old_term = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                self.get_logger().warn("Keyboard E-STOP enabled: SPACE=STOP(latch), r=RELEASE  ※この端末がアクティブの時のみ有効")
            except Exception as e:
                self.get_logger().warn(f"Keyboard E-STOP init failed: {e}")
                self._kb_old_term = None

            self._kb_timer = self.create_timer(0.05, self._kb_poll)

        self.get_logger().info("safety_interface started (optical-aware, STOP min-hold + rapid-approach STOP enabled)")

    # ---------------- callbacks ----------------
    def cb_cmd(self, msg: Twist):
        self.latest_cmd = msg

    def cb_tracked(self, msg: MarkerArray):
        pts = [self._pose_to_base(m) for m in msg.markers]
        if pts:
            self.recent_tracked.append((time.time(), pts))

    def cb_static(self, msg: MarkerArray):
        self._last_static_time = time.time()
        self.static_objs = [self._pose_to_base(m) for m in msg.markers]

    def cb_intent(self, msg: String):
        parts = msg.data.split(':')
        if len(parts) < 2:
            return
        state = parts[1]
        if 'crossing' not in state:
            return

        self._last_cross_time = time.time()
        if len(parts) >= 7:
            try:
                self._last_cross_rng = float(parts[6])
            except Exception:
                self._last_cross_rng = float('nan')
        else:
            self._last_cross_rng = float('nan')

        if (not math.isnan(self._last_cross_rng)) and (float(self._last_cross_rng) >= float(self.cross_slow_dist)):
            self._far_cross_until = max(self._far_cross_until, time.time() + float(self.far_cross_suppress_sec))

    # ---------------- keyboard estop ----------------
    def _kb_poll(self):
        if self._kb_old_term is None:
            return
        try:
            if select.select([sys.stdin], [], [], 0.0)[0]:
                ch = sys.stdin.read(1)
                if ch == ' ':
                    if not self._kb_estop_latched:
                        self.get_logger().warn("E-STOP LATCHED (SPACE)")
                    self._kb_estop_latched = True
                elif ch == 'r':
                    if self._kb_estop_latched:
                        self.get_logger().warn("E-STOP RELEASED (r)")
                    self._kb_estop_latched = False
        except Exception:
            pass

    # ---------------- transform ----------------
    def _pose_to_base(self, marker):
        if TF2_AVAILABLE and marker.header.frame_id:
            try:
                ps = PoseStamped()
                ps.header = marker.header
                ps.pose = marker.pose
                tr = self.tf_buffer.lookup_transform(
                    self.base_frame, marker.header.frame_id, rclpy.time.Time())
                out = tf2_geometry_msgs.do_transform_pose(ps, tr)
                p = out.pose.position
                return (float(p.x), float(p.y), float(p.z))
            except Exception:
                pass

        return (float(marker.pose.position.x),
                float(marker.pose.position.y),
                float(marker.pose.position.z))

    # ---------------- helpers ----------------
    def _closest_dynamic_front(self):
        now = time.time()
        dmin = None

        z_min = float(self.dyn_gate_x_min)
        z_max = float(self.dyn_gate_x_max)
        x_half = float(self.dyn_gate_y_half)

        for t, pts in self.recent_tracked:
            if now - t > 1.2:
                continue
            for x, y, z in pts:
                if z <= z_min:
                    continue
                if z >= z_max:
                    continue
                if abs(x) >= x_half:
                    continue
                d = z
                dmin = d if dmin is None else min(dmin, d)

        return dmin

    def _static_effective_objs(self):
        if (time.time() - self._last_static_time) <= float(self.static_cache_sec):
            return self.static_objs
        return []

    def _closest_static_front(self):
        objs = self._static_effective_objs()
        if not objs:
            return None

        dmin = None
        z_max = float(self.emg_gate_x_max)
        x_half = float(self.emg_gate_y_half)

        for x, y, z in objs:
            if z <= 0.0:
                continue
            if z > z_max:
                continue
            if abs(x) > x_half:
                continue
            d = z
            dmin = d if dmin is None else min(dmin, d)

        return dmin

    def _closest_dynamic_front_emg(self):
        now = time.time()
        dmin = None
        z_max = float(self.emg_gate_x_max)
        x_half = float(self.emg_gate_y_half)

        for t, pts in self.recent_tracked:
            if now - t > 1.0:
                continue
            for x, y, z in pts:
                if z <= 0.0:
                    continue
                if z > z_max:
                    continue
                if abs(x) > x_half:
                    continue
                d = z
                dmin = d if dmin is None else min(dmin, d)

        return dmin

    def _slew_v(self, v_target: float):
        if not bool(self.use_slew_limiter):
            self._v_out = v_target
            return v_target

        dt = float(self.tick_dt)
        dv_up = float(self.max_accel) * dt
        dv_dn = float(self.max_decel) * dt

        v = float(self._v_out)
        if v_target > v:
            v = min(v + dv_up, v_target)
        else:
            v = max(v - dv_dn, v_target)
        self._v_out = v
        return v

    def _publish(self, v, w):
        tw = Twist()
        tw.linear.x = self._slew_v(max(0.0, float(v)))
        tw.angular.z = float(w)
        self.pub_cmd.publish(tw)

    def _stop(self):
        self._publish(0.0, 0.0)

    def _hold_sec_for_mode(self, mode: str, hold_sec: float) -> float:
        if mode == "STOP":
            return max(float(hold_sec), float(self.stop_min_hold_sec))
        return float(hold_sec)

    # ★追加：reason の “比較用キー” を作る（距離など数値変動を無視）
    def _reason_key(self, reason: str) -> str:
        # "APPROACH_SLOW d=0.71" -> "APPROACH_SLOW"
        # "CROSS_SLOW rng=0.72"  -> "CROSS_SLOW"
        # "EMERGENCY_STOP d=.."  -> "EMERGENCY_STOP"
        try:
            return str(reason).split()[0]
        except Exception:
            return str(reason)

    def _set_hold(self, mode: str, reason: str, hold_sec: float):
        """
        holdの仕様：
        - STOP は最低 stop_min_hold_sec を必ず満たす
        - STOP は「危険が継続しているなら延長してよい」（安全寄り）
        - SLOW は同“種類”で毎tick延長しない（振動防止）
          ※ "APPROACH_SLOW d=0.74" のように距離が変わっても同じ扱いにする
        """
        now = time.time()
        hold_sec = self._hold_sec_for_mode(mode, hold_sec)
        new_until = now + hold_sec
        key = self._reason_key(reason)

        if mode == "STOP":
            # STOPは安全のため延長を許可（ただし短縮はしない）
            if self._dyn_mode == "STOP":
                if new_until > float(self._dyn_lock_until):
                    self._dyn_lock_until = new_until
                    self._dyn_reason = reason
                    self._dyn_reason_key = key
                return

        # それ以外（SLOW/NORMALなど）は、
        # 同じ mode + 同じ reason_key で lock中なら延長しない（距離の揺れで再装填しない）
        if (self._dyn_mode == mode) and (self._dyn_reason_key == key) and (now < float(self._dyn_lock_until)):
            return

        self._dyn_mode = mode
        self._dyn_reason = reason          # 表示用はそのまま（距離入りOK）
        self._dyn_reason_key = key         # 比較用は距離抜き
        self._dyn_lock_until = new_until

    # ---------------- RAPID APPROACH (rate-based) ----------------
    def _rapid_approach_check(self, dmin):
        if not bool(getattr(self, "rapid_close_enabled", True)):
            return (False, 0.0)

        now = time.time()

        def _prune():
            w = float(getattr(self, "rapid_close_window_sec", 0.5))
            self._dmin_hist = deque([(t, d) for (t, d) in self._dmin_hist if now - t <= w], maxlen=30)

        if dmin is None:
            _prune()
            return (False, 0.0)

        d = float(dmin)
        if d > float(getattr(self, "rapid_close_dist_max", 1.6)):
            _prune()
            return (False, 0.0)

        self._dmin_hist.append((now, d))
        _prune()

        pts = list(self._dmin_hist)
        if len(pts) < int(getattr(self, "rapid_close_min_samples", 3)):
            return (False, 0.0)

        t0, d0 = pts[0]
        t1, d1 = pts[-1]
        dt = float(t1 - t0)
        if dt <= 1e-3:
            return (False, 0.0)

        v_close = (float(d0) - float(d1)) / dt
        if v_close >= float(getattr(self, "rapid_close_v_thresh", 1.0)):
            return (True, float(v_close))

        return (False, float(v_close))

    # ---------------- STATIC latch ----------------
    def _static_raw_on(self, objs):
        for x, y, z in objs:
            if 0.0 < z < float(self.front_trigger_dist) and abs(x) < float(self.path_half_width):
                return True
        return False

    def _static_raw_off(self, objs):
        for x, y, z in objs:
            if 0.0 < z < float(self.front_off_dist) and abs(x) < float(self.path_off_half_width):
                return False
        return True

    def _update_static_latch(self):
        objs = self._static_effective_objs()

        if not self._static_latched:
            self._static_true_cnt = self._static_true_cnt + 1 if self._static_raw_on(objs) else 0
            if self._static_true_cnt >= int(self.static_on_frames):
                self._static_latched = True
                self._static_true_cnt = 0
                self._static_false_cnt = 0
        else:
            self._static_false_cnt = self._static_false_cnt + 1 if self._static_raw_off(objs) else 0
            if self._static_false_cnt >= int(self.static_off_frames):
                self._static_latched = False
                self._static_true_cnt = 0
                self._static_false_cnt = 0

        return self._static_latched, objs

    def _free_space_score(self, side, objs):
        score = 0.0
        scan = float(self.lateral_scan_dist)
        for x, y, z in objs:
            if 0 < z < scan:
                if side == 'LEFT' and x < 0:
                    score += 1.0 / max(z, 0.1)
                if side == 'RIGHT' and x > 0:
                    score += 1.0 / max(z, 0.1)
        return score

    # ---------------- main loop ----------------
    def tick(self):
        if bool(getattr(self, "_kb_estop_latched", False)):
            self.pub_state.publish(String(data="E_STOP_KEYBOARD_LATCHED"))
            self._stop()
            return

        now = time.time()
        lock_active = now < float(self._dyn_lock_until)
        lock_remain = max(0.0, float(self._dyn_lock_until) - now)

        # ===== (0) EMERGENCY STOP (highest priority) =====
        d_dyn_emg = self._closest_dynamic_front_emg()
        d_sta_emg = self._closest_static_front()
        d_any = None
        if d_dyn_emg is not None:
            d_any = d_dyn_emg
        if d_sta_emg is not None:
            d_any = d_sta_emg if d_any is None else min(d_any, d_sta_emg)

        if d_any is not None and d_any < float(self.emergency_stop_dist):
            self._set_hold("STOP", f"EMERGENCY_STOP d={d_any:.2f}", float(self.emergency_hold_sec))
            lock_remain = max(0.0, float(self._dyn_lock_until) - now)
            self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
            self._stop()
            return

        # ===== (A) CROSSING near-only =====
        cross_recent = (now - float(self._last_cross_time)) < float(self.cross_cache_sec)
        if cross_recent and not math.isnan(self._last_cross_rng):
            rng = float(self._last_cross_rng)

            if rng < float(self.cross_stop_dist):
                self._set_hold("STOP", f"CROSS_STOP rng={rng:.2f}", float(self.cross_hold_sec))
            elif rng < float(self.cross_slow_dist):
                self._set_hold("SLOW", f"CROSS_SLOW rng={rng:.2f}", float(self.cross_hold_sec))
            else:
                if not lock_active:
                    self._dyn_mode = "NORMAL"
                    self._dyn_reason = f"CROSS_FAR_IGNORE rng={rng:.2f}"
                    self._dyn_reason_key = self._reason_key(self._dyn_reason)
                    self._dyn_lock_until = 0.0

            lock_active = now < float(self._dyn_lock_until)
            lock_remain = max(0.0, float(self._dyn_lock_until) - now)

            if self._dyn_mode == "STOP":
                self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
                self._stop()
                return
            if self._dyn_mode == "SLOW":
                self.pub_state.publish(String(data=f"DYN_SLOW {self._dyn_reason} lock={lock_remain:.2f}s"))
                self._publish(float(self.slow_speed), float(self.latest_cmd.angular.z))
                return

        # ===== (B) APPROACH (dynamic), suppressed during FAR-cross window =====
        if now < float(self._far_cross_until):
            if lock_active:
                lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                if self._dyn_mode == "STOP":
                    self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
                    self._stop()
                    return
                if self._dyn_mode == "SLOW":
                    self.pub_state.publish(String(data=f"DYN_SLOW {self._dyn_reason} lock={lock_remain:.2f}s"))
                    self._publish(float(self.slow_speed), float(self.latest_cmd.angular.z))
                    return

            dmin = self._closest_dynamic_front()
            rapid, v_close = self._rapid_approach_check(dmin)

            if rapid:
                self._set_hold(
                    "STOP",
                    f"RAPID_APPROACH v={v_close:.2f} d={float(dmin):.2f}",
                    float(self.rapid_close_hold_sec)
                )
                lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
                self._stop()
                return

            v_in = float(self.latest_cmd.linear.x)
            v_cmd = v_in if abs(v_in) > 1e-6 else float(self.default_speed)
            self.pub_state.publish(String(data=f"FAR_CROSS_SUPPRESS_APPROACH vclose={v_close:.2f}"))
            self._publish(v_cmd, float(self.latest_cmd.angular.z))
            return
        else:
            dmin = self._closest_dynamic_front()

            rapid, v_close = self._rapid_approach_check(dmin)
            if rapid:
                self._set_hold(
                    "STOP",
                    f"RAPID_APPROACH v={v_close:.2f} d={float(dmin):.2f}",
                    float(self.rapid_close_hold_sec)
                )
                lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
                self._stop()
                return

            if lock_active:
                if self._dyn_mode == "SLOW" and dmin is not None and dmin < float(self.stop_dist):
                    self._set_hold("STOP", f"APPROACH_STOP d={dmin:.2f}", float(self.stop_hold_sec))

                lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                if self._dyn_mode == "STOP":
                    self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
                    self._stop()
                    return
                if self._dyn_mode == "SLOW":
                    self.pub_state.publish(String(data=f"DYN_SLOW {self._dyn_reason} lock={lock_remain:.2f}s"))
                    self._publish(float(self.slow_speed), float(self.latest_cmd.angular.z))
                    return

            if dmin is not None and dmin < float(self.stop_dist):
                self._set_hold("STOP", f"APPROACH_STOP d={dmin:.2f}", float(self.stop_hold_sec))
                lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
                self._stop()
                return

            if dmin is not None and dmin < float(self.slow_dist):
                self._set_hold("SLOW", f"APPROACH_SLOW d={dmin:.2f}", float(self.slow_hold_sec))
                lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                self.pub_state.publish(String(data=f"DYN_SLOW {self._dyn_reason} lock={lock_remain:.2f}s"))
                self._publish(float(self.slow_speed), float(self.latest_cmd.angular.z))
                return

        # ===== (C) STATIC avoid + extra hold =====
        static_on, objs = self._update_static_latch()
        if static_on:
            self._static_avoid_until = max(self._static_avoid_until, now + float(self.static_avoid_hold_sec))

        if static_on or (now < float(self._static_avoid_until)):
            objs_eff = objs if objs else self._static_effective_objs()
            left = self._free_space_score('LEFT', objs_eff)
            right = self._free_space_score('RIGHT', objs_eff)
            self.pub_state.publish(String(data=f"STATIC_AVOID L:{left:.2f} R:{right:.2f} hold={(max(0.0,self._static_avoid_until-now)):.2f}s"))

            if left < right:
                self._publish(0.0, +float(self.avoidance_angular))
            else:
                self._publish(0.0, -float(self.avoidance_angular))
            return

        # ===== (D) NORMAL =====
        v_in = float(self.latest_cmd.linear.x)
        v_cmd = v_in if abs(v_in) > 1e-6 else float(self.default_speed)
        self.pub_state.publish(String(data="NORMAL"))
        self._publish(v_cmd, float(self.latest_cmd.angular.z))

    def start(self):
        self.create_timer(float(self.tick_dt), self.tick)


def main():
    rclpy.init()
    node = SafetyInterface()
    node.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._stop()
            time.sleep(0.1)
        except Exception:
            pass

        try:
            if getattr(node, "_kb_old_term", None) is not None:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, node._kb_old_term)
        except Exception:
            pass

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
