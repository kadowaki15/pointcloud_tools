#!/usr/bin/env python3
"""
safety_interface (ROS2 Humble) - FULL VERSION (intent + robust rapid)

前提（あなたの今の環境）:
- /tf は publisher 0（動的TFなし）
- /tf_static は RealSense が出している（静的TFのみ）
- /tracked_markers は frame_id = camera_depth_optical_frame（optical座標）
  optical座標の軸: 前方=z、左右=x（ここでは z を前方距離として扱う）

この版の狙い:
- RViz(=intent_estimatorのTEXT)で approach/approach_fast が出たら safety_state も追随する
- ただし “横切りでちょっと近づく” 程度では rapid_stop(急接近STOP)が出ないように絞る
- STOP lock中は他判定で絶対に上書きさせない（二重ガード）
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
        self.declare_parameter('emg_gate_x_max', 0.80)   # = z_max
        self.declare_parameter('emg_gate_y_half', 0.45)  # = x_half

        # ---------------- APPROACH (distance-based) ----------------
        self.declare_parameter('stop_dist', 0.45)
        self.declare_parameter('slow_dist', 1.00)
        self.declare_parameter('stop_hold_sec', 1.5)
        self.declare_parameter('slow_hold_sec', 1.2)

        # approach gate（optical前提: z_min/z_max として解釈）
        self.declare_parameter('dyn_gate_x_min', 0.10)  # = z_min
        self.declare_parameter('dyn_gate_x_max', 2.00)  # = z_max
        self.declare_parameter('dyn_gate_y_half', 0.45) # = x_half

        # ---------------- CROSSING (intent-based only) ----------------
        self.declare_parameter('cross_stop_dist', 0.35)
        self.declare_parameter('cross_slow_dist', 0.80)
        self.declare_parameter('cross_hold_sec', 0.6)
        self.declare_parameter('cross_cache_sec', 0.8)

        # FAR-cross suppress (to avoid “far crossing -> approach slow/stop”)
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

        # ---------------- INTENT (approach / approach_fast) ----------------
        # intentを拾う時間（RViz表示に追随させるため）
        self.declare_parameter('intent_cache_sec', 0.6)
        # approach_fastは遠すぎる時は無視（横切りで誤爆しやすい距離を切る）
        self.declare_parameter('intent_fast_dist_max', 1.2)
        # approach（通常）の場合、距離でSLOW/STOPにする
        self.declare_parameter('intent_use_approach', True)

        # ---------------- RAPID APPROACH (robust) ----------------
        self.declare_parameter('rapid_close_enabled', True)

        # (A) “接近速度”閾値（m/s）
        self.declare_parameter('rapid_close_v_thresh', 1.3)

        # (B) 観測窓（sec）とサンプル数
        self.declare_parameter('rapid_close_window_sec', 0.35)
        self.declare_parameter('rapid_close_min_samples', 4)

        # (C) rapidを見る距離レンジ
        self.declare_parameter('rapid_close_dist_max', 1.2)      # これより遠い d はrapid対象外
        self.declare_parameter('rapid_close_dist_trigger', 1.0)  # これより近い領域でのみrapid許可

        # (D) “ちょっと近づいた”を殺す（総減少量）
        self.declare_parameter('rapid_close_drop_min', 0.25)     # dがこの量以上詰まったら本物扱い

        # (E) “横切り”を殺す（中心線・横速度・方向比）
        self.declare_parameter('rapid_close_x_center', 0.15)     # |x|がこれ以内の時だけrapid許可
        self.declare_parameter('rapid_close_lat_v_max', 0.20)    # 横速度|vx_lat|がこれ超えたらrapid無効
        self.declare_parameter('rapid_close_dir_ratio_min', 6.0) # v_close / |v_lat| がこれ以上でのみrapid許可

        # hold
        self.declare_parameter('rapid_close_hold_sec', 1.0)

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
        self.recent_tracked = deque(maxlen=12)   # (t, pts)

        # rapid history: (t, dmin, x_at_dmin)
        self._dmin_hist = deque(maxlen=40)

        # static
        self.static_objs = []
        self._last_static_time = 0.0
        self._static_latched = False
        self._static_true_cnt = 0
        self._static_false_cnt = 0
        self._static_avoid_until = 0.0

        # intent cache
        self._last_cross_time = 0.0
        self._last_cross_rng = float('nan')
        self._far_cross_until = 0.0

        self._last_approach_time = 0.0
        self._last_approach_state = "unknown"   # approach / approach_fast
        self._last_approach_dist = float('nan') # prefer z if available

        # dynamic hold
        self._dyn_mode = "NORMAL"      # NORMAL / SLOW / STOP
        self._dyn_lock_until = 0.0
        self._dyn_reason = "NORMAL"
        self._dyn_reason_key = "NORMAL"  # distance-less key

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

        self.get_logger().info("safety_interface started (intent + robust rapid + STOP lock priority)")

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
        """
        受け取り形式に耐える（どれが来ても落ちない）
        例:
          old : id:state:fwd:vx:vz:z
          new : id:state:vr:vt:vx:vz:r:z
        """
        parts = msg.data.split(':')
        if len(parts) < 2:
            return

        state = parts[1].strip()
        now = time.time()

        # ---- crossing ----
        if 'crossing' in state:
            self._last_cross_time = now

            rng = float('nan')
            try:
                # new format: ...:r:z  (len>=8)
                if len(parts) >= 8:
                    rng = float(parts[6])
                # some other formats: last is z
                elif len(parts) >= 3:
                    rng = float(parts[-1])
            except Exception:
                rng = float('nan')

            if not math.isnan(rng):
                # 負値はノイズとして0に丸め
                rng = max(0.0, float(rng))
            self._last_cross_rng = rng

            # FAR-cross suppress: “遠い横切り”は approach 側の誤検出を抑えたい
            if (not math.isnan(self._last_cross_rng)) and (float(self._last_cross_rng) >= float(self.cross_slow_dist)):
                self._far_cross_until = max(self._far_cross_until, now + float(self.far_cross_suppress_sec))
            return

        # ---- approach / approach_fast ----
        if state.startswith('approach'):
            self._last_approach_time = now
            self._last_approach_state = state

            # dist: new format has last z, old format has last z
            dist = float('nan')
            try:
                dist = float(parts[-1])
            except Exception:
                dist = float('nan')

            if not math.isnan(dist):
                # negative is noise
                dist = max(0.0, float(dist))
            self._last_approach_dist = dist
            return

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
        """
        return (dmin_z, x_at_dmin)
        """
        now = time.time()
        dmin = None
        xmin = 0.0

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
                d = float(z)
                if dmin is None or d < dmin:
                    dmin = d
                    xmin = float(x)

        return dmin, xmin

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
            d = float(z)
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
                d = float(z)
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
        """
        holdの仕様：
        - STOP は最低 stop_min_hold_sec を必ず満たす
        - STOP lock中に SLOW/NORMALへ格下げは絶対しない（安全優先）
        - STOP は危険が継続しているなら延長OK
        - SLOW は同“種類”で毎tick延長しない（振動防止）
        """
        now = time.time()

        # STOP lock中は格下げ禁止（最重要）
        if self._dyn_mode == "STOP" and (now < float(self._dyn_lock_until)) and mode != "STOP":
            return

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

    # ---------------- RAPID APPROACH (robust) ----------------
    def _rapid_approach_check(self, dmin, x_at_dmin):
        """
        v_close = (d_old - d_new)/dt が大きい => 急接近
        ただし横切りで “少し近づく成分” は必ず出るので、
        以下の追加条件で「本当に危険」だけに絞る。

        Gate:
          - d <= rapid_close_dist_max
          - last d <= rapid_close_dist_trigger（近い領域でしか許可しない）
          - total drop >= rapid_close_drop_min（ちょっと近づきは無視）
          - |x| <= rapid_close_x_center（中心線近傍のみ）
          - |v_lat| <= rapid_close_lat_v_max（横移動強ければ無効）
          - v_close / (|v_lat|+eps) >= rapid_close_dir_ratio_min
        """
        if not bool(getattr(self, "rapid_close_enabled", True)):
            return (False, 0.0, 0.0, 0.0)  # (rapid, v_close, v_lat, drop)

        now = time.time()

        def _prune():
            w = float(getattr(self, "rapid_close_window_sec", 0.35))
            self._dmin_hist = deque([(t, d, x) for (t, d, x) in self._dmin_hist if now - t <= w], maxlen=40)

        if dmin is None:
            _prune()
            return (False, 0.0, 0.0, 0.0)

        d = float(dmin)
        if d > float(getattr(self, "rapid_close_dist_max", 1.2)):
            _prune()
            return (False, 0.0, 0.0, 0.0)

        self._dmin_hist.append((now, d, float(x_at_dmin)))
        _prune()

        pts = list(self._dmin_hist)
        if len(pts) < int(getattr(self, "rapid_close_min_samples", 4)):
            return (False, 0.0, 0.0, 0.0)

        # use oldest/newest
        t0, d0, x0 = pts[0]
        t1, d1, x1 = pts[-1]
        dt = float(t1 - t0)
        if dt <= 1e-3:
            return (False, 0.0, 0.0, 0.0)

        drop = float(d0 - d1)         # + => closer
        v_close = drop / dt           # + => approaching fast
        v_lat = (float(x1) - float(x0)) / dt

        # ---- filters to kill “slight diagonal approach” ----
        # (1) near-zone only
        if float(d1) > float(getattr(self, "rapid_close_dist_trigger", 1.0)):
            return (False, float(v_close), float(v_lat), float(drop))

        # (2) minimum drop
        if float(drop) < float(getattr(self, "rapid_close_drop_min", 0.25)):
            return (False, float(v_close), float(v_lat), float(drop))

        # (3) centerline only
        if abs(float(x1)) > float(getattr(self, "rapid_close_x_center", 0.15)):
            return (False, float(v_close), float(v_lat), float(drop))

        # (4) lateral speed max
        if abs(float(v_lat)) > float(getattr(self, "rapid_close_lat_v_max", 0.20)):
            return (False, float(v_close), float(v_lat), float(drop))

        # (5) direction ratio
        eps = 1e-3
        ratio = float(v_close) / (abs(float(v_lat)) + eps)
        if ratio < float(getattr(self, "rapid_close_dir_ratio_min", 6.0)):
            return (False, float(v_close), float(v_lat), float(drop))

        # final threshold
        if float(v_close) >= float(getattr(self, "rapid_close_v_thresh", 1.3)):
            return (True, float(v_close), float(v_lat), float(drop))

        return (False, float(v_close), float(v_lat), float(drop))

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

        # ===== (0.5) STOP lock中は他の判定で上書きさせない（あなたが聞いたやつ）=====
        if lock_active and self._dyn_mode == "STOP":
            lock_remain = max(0.0, float(self._dyn_lock_until) - now)
            self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
            self._stop()
            return

        # ===== (1) INTENT: approach_fast / approach を拾って反映 =====
        intent_recent = (now - float(self._last_approach_time)) < float(self.intent_cache_sec)
        if intent_recent and bool(self.intent_use_approach):
            st = str(self._last_approach_state)
            d_int = float(self._last_approach_dist) if (not math.isnan(self._last_approach_dist)) else float('nan')

            if st == "approach_fast":
                # 遠すぎる approach_fast は無視（横切り誤爆を抑える）
                if (not math.isnan(d_int)) and (d_int <= float(self.intent_fast_dist_max)):
                    self._set_hold("STOP", f"INTENT_APPROACH_FAST d={d_int:.2f}", float(self.rapid_close_hold_sec))
                    lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                    self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
                    self._stop()
                    return

            elif st == "approach":
                # 距離で通常のSLOW/STOP
                if (not math.isnan(d_int)) and (d_int < float(self.stop_dist)):
                    self._set_hold("STOP", f"INTENT_APPROACH_STOP d={d_int:.2f}", float(self.stop_hold_sec))
                    lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                    self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
                    self._stop()
                    return
                if (not math.isnan(d_int)) and (d_int < float(self.slow_dist)):
                    self._set_hold("SLOW", f"INTENT_APPROACH_SLOW d={d_int:.2f}", float(self.slow_hold_sec))
                    lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                    self.pub_state.publish(String(data=f"DYN_SLOW {self._dyn_reason} lock={lock_remain:.2f}s"))
                    self._publish(float(self.slow_speed), float(self.latest_cmd.angular.z))
                    return

        # ===== (2) CROSSING (intent cache) near-only =====
        cross_recent = (now - float(self._last_cross_time)) < float(self.cross_cache_sec)
        if cross_recent and not math.isnan(self._last_cross_rng):
            rng = float(self._last_cross_rng)

            if rng < float(self.cross_stop_dist):
                self._set_hold("STOP", f"CROSS_STOP rng={rng:.2f}", float(self.cross_hold_sec))
            elif rng < float(self.cross_slow_dist):
                self._set_hold("SLOW", f"CROSS_SLOW rng={rng:.2f}", float(self.cross_hold_sec))
            else:
                # far crossing: speed changeなし
                pass

            if self._dyn_mode == "STOP":
                lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
                self._stop()
                return
            if self._dyn_mode == "SLOW":
                lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                self.pub_state.publish(String(data=f"DYN_SLOW {self._dyn_reason} lock={lock_remain:.2f}s"))
                self._publish(float(self.slow_speed), float(self.latest_cmd.angular.z))
                return

        # ===== (3) dynamic: rapid check + distance-based approach =====
        dmin, x_at_dmin = self._closest_dynamic_front()

        rapid, v_close, v_lat, drop = self._rapid_approach_check(dmin, x_at_dmin)

        # far-cross suppress中は “ちょっと近づく” で遅くならないようにする
        suppress = now < float(self._far_cross_until)

        if rapid:
            # 急接近は常にSTOP（suppress中でも止める）
            dd = float(dmin) if dmin is not None else float('nan')
            self._set_hold(
                "STOP",
                f"RAPID_APPROACH v={v_close:.2f} lat={v_lat:.2f} drop={drop:.2f} d={dd:.2f}",
                float(self.rapid_close_hold_sec)
            )
            lock_remain = max(0.0, float(self._dyn_lock_until) - now)
            self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
            self._stop()
            return

        # suppress中は「本当に近いSTOPだけ許す」(SLOWは原則しない)
        if suppress:
            if dmin is not None and float(dmin) < float(self.stop_dist):
                self._set_hold("STOP", f"APPROACH_STOP d={float(dmin):.2f}", float(self.stop_hold_sec))
                lock_remain = max(0.0, float(self._dyn_lock_until) - now)
                self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
                self._stop()
                return

            # 何もしない（速度は通常）
            v_in = float(self.latest_cmd.linear.x)
            v_cmd = v_in if abs(v_in) > 1e-6 else float(self.default_speed)
            self.pub_state.publish(String(data="FAR_CROSS_SUPPRESS_APPROACH"))
            self._publish(v_cmd, float(self.latest_cmd.angular.z))
            return

        # suppressでない通常時：距離でSLOW/STOP
        if dmin is not None and float(dmin) < float(self.stop_dist):
            self._set_hold("STOP", f"APPROACH_STOP d={float(dmin):.2f}", float(self.stop_hold_sec))
            lock_remain = max(0.0, float(self._dyn_lock_until) - now)
            self.pub_state.publish(String(data=f"DYN_STOP {self._dyn_reason} lock={lock_remain:.2f}s"))
            self._stop()
            return

        if dmin is not None and float(dmin) < float(self.slow_dist):
            self._set_hold("SLOW", f"APPROACH_SLOW d={float(dmin):.2f}", float(self.slow_hold_sec))
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
            self.pub_state.publish(String(data=f"STATIC_AVOID L:{left:.2f} R:{right:.2f} hold={(max(0.0,self._static_avoid_until-now)):.2f}s"))

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
