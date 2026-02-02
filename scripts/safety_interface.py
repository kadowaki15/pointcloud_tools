#!/usr/bin/env python3
"""
safety_interface (ROS2 Humble)

前提（あなたの今の環境）:
- /tf は publisher 0（動的TFなし）
- /tf_static は RealSense が出している（静的TFのみ）
- /tracked_markers は frame_id = camera_depth_optical_frame（optical座標）

この版の狙い:
- /object_intents の approach / approach_fast / crossing を safety 側でも確実に拾う
  /object_intents フォーマット: "id:state:vr:vt:vx:vz:r:z"
- 横切り (vt大) を急接近STOPにしにくい（vt上限 + vr/vt比ゲート）
- far crossing の最中は「普通のapproach減速」は抑制しつつ、
  近距離STOPと approach_fast STOP は通す（安全側）

安全状態出力:
- /safety_state に String を publish
- /cmd_vel_safe に Twist を publish
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
        self.declare_parameter('emg_gate_z_max', 0.80)
        self.declare_parameter('emg_gate_x_half', 0.45)

        # ---------------- APPROACH (dynamic distance gate) ----------------
        # tracked_markers の前方最短 z に基づく最終安全（intent が壊れても止まる）
        self.declare_parameter('stop_dist', 0.45)
        self.declare_parameter('slow_dist', 1.00)
        self.declare_parameter('stop_hold_sec', 1.5)
        self.declare_parameter('slow_hold_sec', 1.2)

        # tracked_markers を見るゲート（optical: z前方, x左右）
        self.declare_parameter('dyn_gate_z_min', 0.10)
        self.declare_parameter('dyn_gate_z_max', 2.00)
        self.declare_parameter('dyn_gate_x_half', 0.45)

        # ---------------- INTENT-based APPROACH ----------------
        # /object_intents の approach/approach_fast を safety_state に反映する
        self.declare_parameter('intent_cache_sec', 0.6)
        self.declare_parameter('intent_use_r_for_cross', True)   # crossing距離は r を使う
        self.declare_parameter('intent_use_z_for_approach', True) # approach距離は z を使う（無ければr）

        # approach: 遠ければ原則無視、近ければ STOP
        self.declare_parameter('intent_approach_slow_dist', 1.0)  # ここより遠いapproachは基本無視
        self.declare_parameter('intent_approach_stop_dist', 0.55) # approachでこの距離未満ならSTOP（tracked側stop_distと整合取りやすい）
        self.declare_parameter('intent_approach_slow_hold_sec', 0.8)
        self.declare_parameter('intent_approach_stop_hold_sec', 1.2)

        # approach_fast: “急接近STOP”
        self.declare_parameter('intent_fast_enabled', True)
        self.declare_parameter('intent_fast_r_max', 1.8)         # 遠距離のfastは無視
        self.declare_parameter('intent_fast_vr_min', 0.35)       # vrがこれ未満ならfast扱いしない（誤爆減）
        self.declare_parameter('intent_fast_vt_max', 0.20)       # vtがこれ超なら横切り寄り => fast無効
        self.declare_parameter('intent_fast_dir_ratio_min', 4.0) # vr/(vt+eps) がこれ未満ならfast無効
        self.declare_parameter('intent_fast_hold_sec', 1.0)

        # ---------------- CROSSING ----------------
        # crossing は intent の距離(r or z)を使って near-only で STOP/SLOW
        self.declare_parameter('cross_stop_dist', 0.35)
        self.declare_parameter('cross_slow_dist', 0.80)
        self.declare_parameter('cross_hold_sec', 0.6)
        self.declare_parameter('cross_cache_sec', 0.8)
        self.declare_parameter('far_cross_suppress_sec', 0.8)

        # far-cross suppress中に許可する最終安全STOP距離（これ未満なら抑制を無視してSTOP）
        self.declare_parameter('suppress_override_stop_dist', 0.55)

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

        # ---------------- Robust intent parsing ----------------
        self.declare_parameter('reject_nonpositive_dist', True)
        self.declare_parameter('min_valid_dist', 0.05)

        # load parameters (ROS2 internal)
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

        # static
        self.static_objs = []
        self._last_static_time = 0.0
        self._static_latched = False
        self._static_true_cnt = 0
        self._static_false_cnt = 0
        self._static_avoid_until = 0.0

        # crossing cache (intent)
        self._last_cross_time = 0.0
        self._last_cross_rng = float('nan')
        self._far_cross_until = 0.0

        # approach cache (intent)
        self._intent_last = {}  # tid -> dict(t, state, vr, vt, r, z)
        self._last_approach_time = 0.0
        self._last_approach_dist = float('nan')
        self._last_fast_time = 0.0
        self._last_fast_dist = float('nan')
        self._last_fast_vr = 0.0
        self._last_fast_vt = 0.0

        # dynamic hold
        self._dyn_mode = "NORMAL"      # NORMAL / SLOW / STOP
        self._dyn_lock_until = 0.0
        self._dyn_reason = "NORMAL"
        self._dyn_reason_key = "NORMAL"

        # slew output
        self._v_out = 0.0

        # Keyboard E-STOP
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

        self.get_logger().info("safety_interface started (intent-aware: crossing/approach/approach_fast + robust parsing)")

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

    def _safe_float(self, s, default=float('nan')):
        try:
            return float(s)
        except Exception:
            return default

    def cb_intent(self, msg: String):
        """
        /object_intents:
          "id:state:vr:vt:vx:vz:r:z"
        state: crossing / approach / approach_fast / static / unknown etc
        """
        parts = msg.data.split(':')
        if len(parts) < 2:
            return

        tid = None
        try:
            tid = int(parts[0])
        except Exception:
            tid = None

        state = str(parts[1]).strip()
        now = time.time()

        # 8項目想定。足りなくても末尾から取れるようにする。
        vr = self._safe_float(parts[2]) if len(parts) > 2 else float('nan')
        vt = self._safe_float(parts[3]) if len(parts) > 3 else float('nan')
        r  = self._safe_float(parts[6]) if len(parts) > 6 else self._safe_float(parts[-2]) if len(parts) >= 2 else float('nan')
        z  = self._safe_float(parts[7]) if len(parts) > 7 else self._safe_float(parts[-1])

        # dist choice
        cross_dist = r if bool(self.intent_use_r_for_cross) else z
        app_dist   = z if bool(self.intent_use_z_for_approach) else r
        if math.isnan(app_dist):
            app_dist = r

        # reject broken distances
        if bool(self.reject_nonpositive_dist):
            if (not math.isnan(cross_dist)) and (cross_dist <= float(self.min_valid_dist)):
                # crossing距離が壊れてる（0や負） => 使わない
                cross_dist = float('nan')
            if (not math.isnan(app_dist)) and (app_dist <= float(self.min_valid_dist)):
                app_dist = float('nan')

        if tid is not None:
            self._intent_last[tid] = {'t': now, 'state': state, 'vr': vr, 'vt': vt, 'r': r, 'z': z}

        # cache crossing
        if 'crossing' in state:
            if not math.isnan(cross_dist):
                self._last_cross_time = now
                self._last_cross_rng = float(cross_dist)
                # far crossing を検知したら approach 抑制窓
                if float(cross_dist) >= float(self.cross_slow_dist):
                    self._far_cross_until = max(self._far_cross_until, now + float(self.far_cross_suppress_sec))

        # cache approach
        if state.startswith('approach'):
            if not math.isnan(app_dist):
                self._last_approach_time = now
                self._last_approach_dist = float(app_dist)

        # cache approach_fast
        if state == 'approach_fast':
            if not math.isnan(app_dist):
                self._last_fast_time = now
                self._last_fast_dist = float(app_dist)
                self._last_fast_vr = 0.0 if math.isnan(vr) else float(vr)
                self._last_fast_vt = 0.0 if math.isnan(vt) else float(vt)

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
                tr = self.tf_buffer.lookup_transform(self.base_frame, marker.header.frame_id, rclpy.time.Time())
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
        """
        tracked_markers の点群から、前方ゲート内で最短の z を返す
        """
        now = time.time()
        dmin = None

        z_min = float(self.dyn_gate_z_min)
        z_max = float(self.dyn_gate_z_max)
        x_half = float(self.dyn_gate_x_half)

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

    def _closest_static_front_emg(self):
        """
        static_obstacles の前方ゲート内最短 z
        """
        objs = self._static_effective_objs()
        if not objs:
            return None

        dmin = None
        z_max = float(self.emg_gate_z_max)
        x_half = float(self.emg_gate_x_half)

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
        """
        tracked_markers の前方ゲート内最短 z（emergency用に短い履歴だけ見る）
        """
        now = time.time()
        dmin = None
        z_max = float(self.emg_gate_z_max)
        x_half = float(self.emg_gate_x_half)

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

    def _reason_key(self, reason: str) -> str:
        try:
            return str(reason).split()[0]
        except Exception:
            return str(reason)

    def _set_hold(self, mode: str, reason: str, hold_sec: float):
        now = time.time()
        hold_sec = self._hold_sec_for_mode(mode, hold_sec)
        new_until = now + hold_sec
        key = self._reason_key(reason)

        if mode == "STOP":
            if self._dyn_mode == "STOP":
                if new_until > float(self._dyn_lock_until):
                    self._dyn_lock_until = new_until
                    self._dyn_reason = reason
                    self._dyn_reason_key = key
                return

        if (self._dyn_mode == mode) and (self._dyn_reason_key == key) and (now < float(self._dyn_lock_until)):
            return

        self._dyn_mode = mode
        self._dyn_reason = reason
        self._dyn_reason_key = key
        self._dyn_lock_until = new_until

    def _intent_fast_should_stop(self, now: float) -> bool:
        """
        intent approach_fast を STOP に使う（横切り誤爆を抑えるゲート付き）
        """
        if not bool(self.intent_fast_enabled):
            return False
        if (now - float(self._last_fast_time)) > float(self.intent_cache_sec):
            return False

        d = float(self._last_fast_dist)
        if math.isnan(d) or d <= float(self.min_valid_dist):
            return False
        if d > float(self.intent_fast_r_max):
            return False

        vr = float(self._last_fast_vr)
        vt = float(self._last_fast_vt)

        if vr < float(self.intent_fast_vr_min):
            return False
        if vt > float(self.intent_fast_vt_max):
            return False

        ratio = vr / max(abs(vt), 1e-3)
        if ratio < float(self.intent_fast_dir_ratio_min):
            return False

        return True

    def _intent_approach_action(self, now: float):
        """
        intent approach を距離で STOP / SLOW / IGNORE に分類
        """
        if (now - float(self._last_approach_time)) > float(self.intent_cache_sec):
            return ("NONE", float('nan'))

        d = float(self._last_approach_dist)
        if math.isnan(d) or d <= float(self.min_valid_dist):
            return ("NONE", d)

        # 遠距離の approach は原則無視（あなたの要望）
        if d > float(self.intent_approach_slow_dist):
            return ("IGNORE", d)

        if d < float(self.intent_approach_stop_dist):
            return ("STOP", d)

        # ここに入るのは「近いが、停止ほどではない」
        return ("SLOW", d)

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
        d_sta_emg = self._closest_static_front_emg()
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

        # ===== (1) INTENT APPROACH_FAST STOP (ゲート付き) =====
        if self._intent_fast_should_stop(now):
            d = float(self._last_fast_dist)
            self._set_hold("STOP", f"INTENT_APPROACH_FAST d={d:.2f}", float(self.intent_fast_hold_sec))
            lock_remain = max(0.0, float(self._dyn_lock_until) - now)
            self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
            self._stop()
            return

        # tracked 最終安全: suppress中でも “近距離はSTOP”
        dmin_track = self._closest_dynamic_front()
        if dmin_track is not None and dmin_track < float(self.suppress_override_stop_dist):
            self._set_hold("STOP", f"TRACK_STOP d={dmin_track:.2f}", float(self.stop_hold_sec))
            lock_remain = max(0.0, float(self._dyn_lock_until) - now)
            self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
            self._stop()
            return

        # ===== (2) CROSSING near-only (intent距離) =====
        cross_recent = (now - float(self._last_cross_time)) < float(self.cross_cache_sec)
        if cross_recent and (not math.isnan(self._last_cross_rng)):
            rng = float(self._last_cross_rng)

            if rng < float(self.cross_stop_dist):
                self._set_hold("STOP", f"CROSS_STOP rng={rng:.2f}", float(self.cross_hold_sec))
            elif rng < float(self.cross_slow_dist):
                self._set_hold("SLOW", f"CROSS_SLOW rng={rng:.2f}", float(self.cross_hold_sec))
            else:
                # far crossing は基本無視（速度変えない）
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

        # ===== (3) APPROACH: far-cross suppress window =====
        suppress = now < float(self._far_cross_until)

        # lock 継続（SLOW/STOP を維持）
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

        # intent approach を反映（ただし suppress中は “遠距離SLOW” を抑制）
        act, d_app = self._intent_approach_action(now)

        if suppress:
            # suppress中: 近距離STOPだけ通す（SLOWは基本抑制）
            if act == "STOP":
                self._set_hold("STOP", f"INTENT_APPROACH_STOP d={d_app:.2f}", float(self.intent_approach_stop_hold_sec))
                lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
                self._stop()
                return

            v_in = float(self.latest_cmd.linear.x)
            v_cmd = v_in if abs(v_in) > 1e-6 else float(self.default_speed)
            self.pub_state.publish(String(data=f"FAR_CROSS_SUPPRESS_APPROACH"))
            self._publish(v_cmd, float(self.latest_cmd.angular.z))
            return
        else:
            # suppress外: intent approach で STOP/SLOW を出す
            if act == "STOP":
                self._set_hold("STOP", f"INTENT_APPROACH_STOP d={d_app:.2f}", float(self.intent_approach_stop_hold_sec))
                lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
                self._stop()
                return
            if act == "SLOW":
                self._set_hold("SLOW", f"INTENT_APPROACH_SLOW d={d_app:.2f}", float(self.intent_approach_slow_hold_sec))
                lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                self.pub_state.publish(String(data=f"DYN_SLOW {self._dyn_reason} lock={lock_remain:.2f}s"))
                self._publish(float(self.slow_speed), float(self.latest_cmd.angular.z))
                return

            # ここまで来たら intent は無視 or そもそも来てない
            # tracked_markers の距離で最終安全をかける（近いならSTOP/SLOW）
            if dmin_track is not None and dmin_track < float(self.stop_dist):
                self._set_hold("STOP", f"TRACK_STOP d={dmin_track:.2f}", float(self.stop_hold_sec))
                lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
                self._stop()
                return
            if dmin_track is not None and dmin_track < float(self.slow_dist):
                self._set_hold("SLOW", f"TRACK_SLOW d={dmin_track:.2f}", float(self.slow_hold_sec))
                lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                self.pub_state.publish(String(data=f"DYN_SLOW {self._dyn_reason} lock={lock_remain:.2f}s"))
                self._publish(float(self.slow_speed), float(self.latest_cmd.angular.z))
                return

        # ===== (4) STATIC avoid + extra hold =====
        static_on, objs = self._update_static_latch()
        if static_on:
            self._static_avoid_until = max(self._static_avoid_until, now + float(self.static_avoid_hold_sec))

        if static_on or (now < float(self._static_avoid_until)):
            objs_eff = objs if objs else self._static_effective_objs()
            left = self._free_space_score('LEFT', objs_eff)
            right = self._free_space_score('RIGHT', objs_eff)
            self.pub_state.publish(String(
                data=f"STATIC_AVOID L:{left:.2f} R:{right:.2f} hold={(max(0.0,self._static_avoid_until-now)):.2f}s"
            ))

            if left < right:
                self._publish(0.0, +float(self.avoidance_angular))
            else:
                self._publish(0.0, -float(self.avoidance_angular))
            return

        # ===== (5) NORMAL =====
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
