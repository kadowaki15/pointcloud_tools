#!/usr/bin/env python3
"""
safety_interface (ROS2 Humble)

KEEP (unchanged):
- APPROACH: distance-based STOP/SLOW + hold (uses dmin)
- STATIC  : avoid (trigger latch to prevent flicker)
- NORMAL  : pass-through cmd_vel (or default_speed)

CHANGE (only crossing):
- CROSSING uses rng (from /object_intents) ONLY
  - far crossing: IGNORE (no slow)
  - near crossing: SLOW/STOP + hold

Extra guard:
- If FAR crossing was seen recently, suppress APPROACH-triggered slow/stop for a short window.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Twist, PoseStamped
import time, math
from collections import deque

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
        self.declare_parameter('default_speed', 0.15)
        self.declare_parameter('slow_speed', 0.05)

        # ---------------- APPROACH (UNCHANGED logic, tuned holds) ----------------
        self.declare_parameter('stop_dist', 0.4)
        self.declare_parameter('slow_dist', 1.0)

        # ★おすすめ（短め）
        self.declare_parameter('stop_hold_sec', 1.2)   # was 2.0
        self.declare_parameter('slow_hold_sec', 1.5)   # was 3.0

        # approach uses dmin but only in front corridor
        self.declare_parameter('dyn_gate_x_min', 0.10)
        self.declare_parameter('dyn_gate_x_max', 2.00)
        self.declare_parameter('dyn_gate_y_half', 0.35)

        # ---------------- CROSSING ----------------
        self.declare_parameter('cross_stop_dist', 0.35)
        self.declare_parameter('cross_slow_dist', 0.80)

        # ★おすすめ（短め）
        self.declare_parameter('cross_hold_sec', 0.6)   # was 1.2

        # crossing label flicker absorber（そのまま）
        self.declare_parameter('cross_cache_sec', 0.8)

        # Guard: far-crossing seen -> suppress approach for short time（そのまま）
        self.declare_parameter('far_cross_suppress_sec', 0.8)

        # ---------------- output smoothing ----------------
        self.declare_parameter('use_slew_limiter', True)
        self.declare_parameter('max_accel', 0.15)  # m/s^2
        self.declare_parameter('max_decel', 0.30)  # m/s^2

        # ---------------- STATIC (UNCHANGED) ----------------
        self.declare_parameter('front_trigger_dist', 0.5)
        self.declare_parameter('lateral_scan_dist', 1.5)
        self.declare_parameter('path_half_width', 0.3)
        self.declare_parameter('avoidance_angular', 0.6)

        self.declare_parameter('static_cache_sec', 0.6)
        self.declare_parameter('static_on_frames', 3)
        self.declare_parameter('static_off_frames', 5)
        self.declare_parameter('front_off_dist', 0.7)
        self.declare_parameter('path_off_half_width', 0.45)

        # ---------------- frames ----------------
        self.declare_parameter('base_frame', 'base_link')

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
        self.sub_static = self.create_subscription(MarkerArray, self.static_topic, self.cb_static, 10)
        self.sub_tracked = self.create_subscription(MarkerArray, self.tracked_topic, self.cb_tracked, 10)
        self.sub_intent = self.create_subscription(String, self.intent_topic, self.cb_intent, 10)
        self.sub_cmd = self.create_subscription(Twist, self.cmd_topic, self.cb_cmd, 10)

        self.pub_cmd = self.create_publisher(Twist, self.publish_topic, 10)
        self.pub_state = self.create_publisher(String, '/safety_state', 10)

        # internal state
        self.latest_cmd = Twist()
        self.recent_tracked = deque(maxlen=10)

        # static
        self.static_objs = []
        self._last_static_time = 0.0
        self._static_latched = False
        self._static_true_cnt = 0
        self._static_false_cnt = 0

        # intent cache for crossing
        self._last_cross_time = 0.0
        self._last_cross_rng = float('nan')
        self._last_cross_kind = ''  # for debug
        self._far_cross_until = 0.0

        # dynamic hold
        self._dyn_mode = "NORMAL"      # NORMAL / SLOW / STOP
        self._dyn_lock_until = 0.0
        self._dyn_reason = "NORMAL"

        # slew output
        self._v_out = 0.0

        self.get_logger().info("safety_interface started (recommended holds + no hold-extension loop)")

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
        # Expected: tid:state:ttc:v_rad:vx:vy:rng
        parts = msg.data.split(':')
        if len(parts) < 2:
            return
        state = parts[1]
        if 'crossing' not in state:
            return

        self._last_cross_time = time.time()
        self._last_cross_kind = state

        if len(parts) >= 7:
            try:
                self._last_cross_rng = float(parts[6])
            except Exception:
                self._last_cross_rng = float('nan')
        else:
            self._last_cross_rng = float('nan')

        # FAR crossing -> suppress approach briefly
        if (not math.isnan(self._last_cross_rng)) and (float(self._last_cross_rng) >= float(self.cross_slow_dist)):
            self._far_cross_until = max(self._far_cross_until, time.time() + float(self.far_cross_suppress_sec))

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
                return (p.x, p.y, p.z)
            except Exception:
                pass
        # fallback mapping (your convention)
        return (marker.pose.position.z, -marker.pose.position.x, -marker.pose.position.y)

    # ---------------- helpers ----------------
    def _closest_dynamic_front(self):
        now = time.time()
        dmin = None

        x_min = float(self.dyn_gate_x_min)
        x_max = float(self.dyn_gate_x_max)
        y_half = float(self.dyn_gate_y_half)

        for t, pts in self.recent_tracked:
            if now - t > 1.2:
                continue
            for x, y, _ in pts:
                if x <= x_min:
                    continue
                if x >= x_max:
                    continue
                if abs(y) >= y_half:
                    continue
                d = math.hypot(x, y)
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

    # ★ここが重要：同じ理由で毎tick延長しない
    def _set_hold(self, mode: str, reason: str, hold_sec: float):
        now = time.time()

        # 同じ mode & reason で、すでに lock 中なら延長しない
        if (self._dyn_mode == mode) and (self._dyn_reason == reason) and (now < float(self._dyn_lock_until)):
            return

        self._dyn_mode = mode
        self._dyn_reason = reason
        self._dyn_lock_until = now + float(hold_sec)

    # ---------------- STATIC latch (unchanged) ----------------
    def _static_effective_objs(self):
        if (time.time() - self._last_static_time) <= float(self.static_cache_sec):
            return self.static_objs
        return []

    def _static_raw_on(self, objs):
        for x, y, _ in objs:
            if 0.0 < x < float(self.front_trigger_dist) and abs(y) < float(self.path_half_width):
                return True
        return False

    def _static_raw_off(self, objs):
        for x, y, _ in objs:
            if 0.0 < x < float(self.front_off_dist) and abs(y) < float(self.path_off_half_width):
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
        for x, y, _ in objs:
            if 0 < x < scan:
                if side == 'LEFT' and y > 0:
                    score += 1.0 / max(x, 0.1)
                if side == 'RIGHT' and y < 0:
                    score += 1.0 / max(x, 0.1)
        return score

    # ---------------- main loop ----------------
    def tick(self):
        now = time.time()
        lock_active = now < float(self._dyn_lock_until)
        lock_remain = max(0.0, float(self._dyn_lock_until) - now)

        # ===== (A) CROSSING near-only =====
        cross_recent = (now - float(self._last_cross_time)) < float(self.cross_cache_sec)
        if cross_recent and not math.isnan(self._last_cross_rng):
            rng = float(self._last_cross_rng)

            if rng < float(self.cross_stop_dist):
                self._set_hold("STOP", f"CROSS_STOP rng={rng:.2f}", float(self.cross_hold_sec))
            elif rng < float(self.cross_slow_dist):
                self._set_hold("SLOW", f"CROSS_SLOW rng={rng:.2f}", float(self.cross_hold_sec))
            else:
                # FAR crossing => ignore
                if not lock_active:
                    self._dyn_mode = "NORMAL"
                    self._dyn_reason = f"CROSS_FAR_IGNORE rng={rng:.2f}"
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
            # NORMAL fall-through

        # ===== (B) APPROACH (unchanged), suppressed during FAR-cross window =====
        if now < float(self._far_cross_until):
            self.pub_state.publish(String(data="FAR_CROSS_SUPPRESS_APPROACH"))
        else:
            dmin = self._closest_dynamic_front()

            # if locked, keep it (only allow SLOW->STOP upgrade)
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

            # not locked: decide fresh
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

        # ===== (C) STATIC (unchanged) =====
        static_on, objs = self._update_static_latch()
        if static_on:
            left = self._free_space_score('LEFT', objs)
            right = self._free_space_score('RIGHT', objs)
            self.pub_state.publish(String(data=f"STATIC_AVOID L:{left:.2f} R:{right:.2f}"))
            if left < right:
                self._publish(0.0, -float(self.avoidance_angular))
            else:
                self._publish(0.0, float(self.avoidance_angular))
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
