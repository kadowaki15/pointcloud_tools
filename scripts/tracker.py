#!/usr/bin/env python3
"""
pointcloud_tools/scripts/tracker.py (ROS2 Humble) - FINAL (GHOST-FREE)

Fixes:
- /dynamic_markers に混ざる Marker.DELETE / DELETEALL を検出扱いしない（←中央ゴーストの根本原因）
- dynamic cluster only: ns == "clusters" だけ追跡（staticや他ns混入対策）
- 不正値/原点付近ゴミを除外（念のため）
- 入力が空でも tracker が変な新規trackを作らない

Other behavior is kept (your original “final” logic).
"""

import os
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String
import numpy as np
from collections import deque


# =================================================
# Simple Kalman Filter (x, y, vx, vy)
# =================================================
class SimpleKalman:
    def __init__(self, dt=0.07, q=1e-2, r=1e-1):
        self.dt = float(dt)
        self.F = np.array(
            [[1, 0, self.dt, 0],
             [0, 1, 0, self.dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)

        self.Q = np.diag([q, q, q * 10, q * 10]).astype(np.float32)
        self.R = np.diag([r, r]).astype(np.float32)
        self.P = np.eye(4, dtype=np.float32) * 0.5
        self.x = np.zeros(4, dtype=np.float32)

    def init(self, x, y):
        self.x[:] = [x, y, 0.0, 0.0]
        self.P = np.eye(4, dtype=np.float32) * 0.5

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, zx, zy):
        z = np.array([zx, zy], dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P

    def pos(self):
        return self.x[:2].copy()

    def vel(self):
        return self.x[2:].copy()


# =================================================
# Tracker Node
# =================================================
class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker")

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter("input_topic", "/dynamic_markers")
        self.declare_parameter("rgb_topic", "/rgb_detections_markers")
        self.declare_parameter("output_topic", "/tracked_markers")
        self.declare_parameter("debug_topic", "/tracked_info")

        # Association
        self.declare_parameter("match_dist", 0.55)
        self.declare_parameter("ttl", 25)
        self.declare_parameter("history_len", 5)
        self.declare_parameter("enable_one_to_one_assoc", True)

        # thresholds (m/s)
        self.declare_parameter("weak_approach_thresh", 0.02)
        self.declare_parameter("strong_approach_thresh", 0.05)

        # CROSSING gate (m/s)
        self.declare_parameter("crossing_thresh", 0.14)
        self.declare_parameter("crossing_ratio", 2.8)

        # APPROACH priority
        self.declare_parameter("approach_priority_fwd", 0.02)
        self.declare_parameter("approach_priority_cnt", 1)

        # RGB match
        self.declare_parameter("rgb_match_dist", 0.6)

        # Z smoothing (EMA)
        self.declare_parameter("z_alpha", 0.35)
        self.declare_parameter("z_default", 0.50)

        # Z outlier gate (meters)
        self.declare_parameter("z_gate", 0.60)

        # dz/dt calculation
        self.declare_parameter("min_dt", 0.02)
        self.declare_parameter("max_dt", 0.20)

        # Ignore tiny motion (noise)
        self.declare_parameter("min_motion", 0.010)

        # clamp insane vx from assoc jumps (m/s)
        self.declare_parameter("vxy_max", 1.2)

        # FSM stability (frames)
        self.declare_parameter("stable_lock_frames", 14)
        self.declare_parameter("confirm_frames", 6)

        # FSM stability (seconds)
        self.declare_parameter("none_hold_sec", 1.0)

        # slow-approach fallback (distance monotonic)
        self.declare_parameter("monotonic_dz_thresh", 0.02)
        self.declare_parameter("monotonic_weak_frames", 3)
        self.declare_parameter("monotonic_strong_frames", 6)

        # -------------------------
        # NEW: input filtering (GHOST FIX)
        # -------------------------
        self.declare_parameter("accept_ns", "clusters")     # dynamic_static_cluster_node dynamic ns
        self.declare_parameter("reject_origin_xy", True)    # drop (0,0) junk
        self.declare_parameter("origin_xy_eps", 0.03)       # 3cm

        # -------------------------
        # Load parameters
        # -------------------------
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)

        self.match_dist = float(self.get_parameter("match_dist").value)
        self.ttl = int(self.get_parameter("ttl").value)
        self.history_len = int(self.get_parameter("history_len").value)
        self.enable_1to1 = bool(self.get_parameter("enable_one_to_one_assoc").value)

        self.weak_th = float(self.get_parameter("weak_approach_thresh").value)
        self.strong_th = float(self.get_parameter("strong_approach_thresh").value)
        self.cross_th = float(self.get_parameter("crossing_thresh").value)
        self.cross_ratio = float(self.get_parameter("crossing_ratio").value)

        self.approach_priority_fwd = float(self.get_parameter("approach_priority_fwd").value)
        self.approach_priority_cnt = int(self.get_parameter("approach_priority_cnt").value)

        self.rgb_match_dist = float(self.get_parameter("rgb_match_dist").value)

        self.z_alpha = float(self.get_parameter("z_alpha").value)
        self.z_default = float(self.get_parameter("z_default").value)
        self.z_gate = float(self.get_parameter("z_gate").value)

        self.min_dt = float(self.get_parameter("min_dt").value)
        self.max_dt = float(self.get_parameter("max_dt").value)
        self.min_motion = float(self.get_parameter("min_motion").value)
        self.vxy_max = float(self.get_parameter("vxy_max").value)

        self.stable_lock_frames = int(self.get_parameter("stable_lock_frames").value)
        self.confirm_frames = int(self.get_parameter("confirm_frames").value)
        self.none_hold_sec = float(self.get_parameter("none_hold_sec").value)

        self.mono_dz = float(self.get_parameter("monotonic_dz_thresh").value)
        self.mono_weak_frames = int(self.get_parameter("monotonic_weak_frames").value)
        self.mono_strong_frames = int(self.get_parameter("monotonic_strong_frames").value)

        self.accept_ns = str(self.get_parameter("accept_ns").value)
        self.reject_origin_xy = bool(self.get_parameter("reject_origin_xy").value)
        self.origin_xy_eps = float(self.get_parameter("origin_xy_eps").value)

        # -------------------------
        # ROS I/O
        # -------------------------
        self.sub = self.create_subscription(MarkerArray, self.input_topic, self.cb, 10)
        self.sub_rgb = self.create_subscription(MarkerArray, self.rgb_topic, self.rgb_cb, 10)

        self.pub = self.create_publisher(MarkerArray, self.output_topic, 10)
        self.pub_dbg = self.create_publisher(String, self.debug_topic, 10)

        # -------------------------
        # Internal state
        # -------------------------
        self.tracks = {}
        self.next_id = 0
        self.frame = 0
        self.latest_rgb = []

        self.get_logger().warn(f"### TRACKER FINAL RUNNING (scripts): {os.path.abspath(__file__)} ###")
        self.get_logger().warn(f"Input filter: accept_ns='{self.accept_ns}', ignore action!=ADD, reject_origin_xy={self.reject_origin_xy}")

    def now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    # -------------------------------------------------
    # RGB callback
    # -------------------------------------------------
    def rgb_cb(self, msg: MarkerArray):
        self.latest_rgb = []
        for m in msg.markers:
            self.latest_rgb.append(np.array([m.pose.position.x, m.pose.position.y], dtype=np.float32))

    # -------------------------------------------------
    # Main callback
    # -------------------------------------------------
    def cb(self, msg: MarkerArray):
        self.frame += 1
        now = self.now_sec()

        # =========================
        # Build detections (FIXED)
        # =========================
        detections = []
        for m in msg.markers:
            # 1) ✅ DELETE/DELETEALL を検出扱いしない（中央ゴースト根絶）
            if m.action != Marker.ADD:
                continue

            # 2) ✅ dynamic only (ns="clusters") を追跡
            if self.accept_ns and m.ns != self.accept_ns:
                continue

            x = float(m.pose.position.x)
            y = float(m.pose.position.y)
            z = float(m.pose.position.z)

            # 3) ✅ NaN/Inf 排除
            if (not np.isfinite(x)) or (not np.isfinite(y)) or (not np.isfinite(z)):
                continue

            # 4) ✅ 原点付近ゴミ排除（念のため）
            if self.reject_origin_xy and (abs(x) < self.origin_xy_eps) and (abs(y) < self.origin_xy_eps):
                # 原点付近は DELETEALL/未初期化/ゴミの典型
                continue

            detections.append(np.array([x, y, z], dtype=np.float32))

        out = MarkerArray()
        debug_lines = []

        # per frame flags
        for tr in self.tracks.values():
            tr["z_updated"] = False

        # ---------- Predict ----------
        for tr in self.tracks.values():
            tr["kf"].predict()
            tr["history"].append(tr["kf"].pos())
            if len(tr["history"]) > self.history_len:
                tr["history"].popleft()

        # ---------- Associate ----------
        used_tracks = set()

        for det in detections:
            det_xy = det[:2]
            det_z = float(det[2])

            best_id, best_d = None, float("inf")
            for tid, tr in self.tracks.items():
                if self.enable_1to1 and tid in used_tracks:
                    continue
                d = float(np.linalg.norm(det_xy - tr["kf"].pos()))
                if d < best_d and d < self.match_dist:
                    best_d, best_id = d, tid

            if best_id is not None:
                used_tracks.add(best_id)
                tr = self.tracks[best_id]
                tr["kf"].update(det_xy[0], det_xy[1])
                tr["last"] = self.frame

                # z update with gate
                if np.isfinite(det_z) and det_z > 0.0:
                    prev = float(tr.get("z", self.z_default))
                    if abs(det_z - prev) <= self.z_gate:
                        tr["z"] = (1.0 - self.z_alpha) * prev + self.z_alpha * det_z
                        tr["z_last"] = det_z
                        tr["z_updated"] = True
                    else:
                        tr["z_last"] = float("nan")
                else:
                    tr["z_last"] = float("nan")

            else:
                kf = SimpleKalman(dt=0.07)
                kf.init(det_xy[0], det_xy[1])

                z_init = det_z if (np.isfinite(det_z) and det_z > 0.0) else self.z_default
                self.tracks[self.next_id] = {
                    "kf": kf,
                    "history": deque([det_xy], maxlen=self.history_len),
                    "last": self.frame,

                    "weak_cnt": 0,
                    "strong_cnt": 0,

                    "z": float(z_init),
                    "z_last": float(det_z),
                    "z_updated": True,

                    "z_prev": float(z_init),
                    "t_prev": float(now),
                    "vz": 0.0,

                    "z_mon_cnt": 0,
                    "z_prev2": float(z_init),

                    "stable_state": "NONE",
                    "stable_group": "NONE",
                    "lock_cnt": 0,
                    "raw_cnt": 0,
                    "last_raw": "NONE",

                    "none_start_t": None,
                }
                self.next_id += 1

        # ---------- Remove old ----------
        for tid in list(self.tracks.keys()):
            if self.frame - self.tracks[tid]["last"] > self.ttl:
                m = Marker()
                m.header.frame_id = "camera_depth_optical_frame"
                m.ns = "tracked"
                m.id = tid
                m.action = Marker.DELETE
                out.markers.append(m)
                del self.tracks[tid]

        # ---------- State estimation & publish ----------
        for tid, tr in self.tracks.items():
            pos = tr["kf"].pos()
            vel = tr["kf"].vel()

            # clamp vxy
            vel[0] = float(np.clip(vel[0], -self.vxy_max, self.vxy_max))
            vel[1] = float(np.clip(vel[1], -self.vxy_max, self.vxy_max))

            rgb_match = any(float(np.linalg.norm(rgb - pos)) < self.rgb_match_dist for rgb in self.latest_rgb)

            # dz/dt only if z updated
            z_now = float(tr.get("z", self.z_default))
            z_prev = float(tr.get("z_prev", z_now))
            t_prev = float(tr.get("t_prev", now))

            dt = now - t_prev
            if not np.isfinite(dt) or dt <= 0.0:
                dt = self.min_dt
            dt = float(np.clip(dt, self.min_dt, self.max_dt))

            if tr.get("z_updated", False):
                vz = (z_now - z_prev) / dt
                tr["z_prev"] = z_now
                tr["t_prev"] = now
            else:
                vz = 0.0
            tr["vz"] = float(vz)

            forward = max(0.0, -vz)      # approach only
            lateral = abs(float(vel[0])) # x axis

            # monotonic fallback
            z_prev2 = float(tr.get("z_prev2", z_now))
            dz2 = z_now - z_prev2
            tr["z_prev2"] = z_now
            if dz2 < -self.mono_dz:
                tr["z_mon_cnt"] = int(tr.get("z_mon_cnt", 0)) + 1
            else:
                tr["z_mon_cnt"] = 0

            # raw state
            raw_state = "NONE"

            if forward + lateral < self.min_motion:
                tr["weak_cnt"] = 0
                tr["strong_cnt"] = 0
                tr["z_mon_cnt"] = 0
                raw_state = "NONE"
            else:
                if forward > self.weak_th:
                    tr["weak_cnt"] += 1
                else:
                    tr["weak_cnt"] = 0

                if forward > self.strong_th:
                    tr["strong_cnt"] += 1
                else:
                    tr["strong_cnt"] = 0

                if tr["strong_cnt"] >= (2 if rgb_match else 3):
                    raw_state = "STRONG_APPROACH"
                elif tr["weak_cnt"] >= 2:
                    raw_state = "WEAK_APPROACH"
                else:
                    if tr.get("z_mon_cnt", 0) >= self.mono_strong_frames:
                        raw_state = "STRONG_APPROACH"
                    elif tr.get("z_mon_cnt", 0) >= self.mono_weak_frames:
                        raw_state = "WEAK_APPROACH"
                    else:
                        raw_state = "NONE"

                # crossing allowed only if not approaching-ish
                crossing_allowed = True
                if forward >= self.approach_priority_fwd:
                    crossing_allowed = False
                if tr.get("weak_cnt", 0) >= self.approach_priority_cnt:
                    crossing_allowed = False
                if tr.get("z_mon_cnt", 0) >= 2:
                    crossing_allowed = False

                if raw_state == "NONE" and crossing_allowed:
                    if (lateral > self.cross_th) and (lateral > self.cross_ratio * max(1e-6, forward)):
                        raw_state = "CROSSING"

            # stable FSM
            def group_of(s: str) -> str:
                if s in ("WEAK_APPROACH", "STRONG_APPROACH"):
                    return "APPROACH"
                if s == "CROSSING":
                    return "CROSSING"
                return "NONE"

            raw_group = group_of(raw_state)

            # raw streak
            if tr.get("last_raw", "NONE") == raw_state:
                tr["raw_cnt"] = tr.get("raw_cnt", 0) + 1
            else:
                tr["raw_cnt"] = 1
                tr["last_raw"] = raw_state

            stable_state = tr.get("stable_state", "NONE")
            stable_group = tr.get("stable_group", "NONE")

            # lock decrement
            if tr.get("lock_cnt", 0) > 0:
                tr["lock_cnt"] -= 1

            # NONE timer
            if raw_group == "NONE":
                if tr.get("none_start_t", None) is None:
                    tr["none_start_t"] = now
            else:
                tr["none_start_t"] = None

            # same-group updates
            if stable_group == "APPROACH" and raw_group == "APPROACH":
                stable_state = raw_state
                stable_group = "APPROACH"
                tr["lock_cnt"] = self.stable_lock_frames

            elif stable_group == "CROSSING" and raw_group == "CROSSING":
                stable_state = "CROSSING"
                stable_group = "CROSSING"
                tr["lock_cnt"] = self.stable_lock_frames

            else:
                # mutual transition forbidden (APPROACH <-> CROSSING)
                if stable_group in ("APPROACH", "CROSSING") and raw_group in ("APPROACH", "CROSSING"):
                    pass
                elif stable_group == "NONE":
                    if raw_group != "NONE" and tr["raw_cnt"] >= self.confirm_frames:
                        stable_group = raw_group
                        stable_state = raw_state if raw_group == "APPROACH" else "CROSSING"
                        tr["lock_cnt"] = self.stable_lock_frames
                else:
                    none_start = tr.get("none_start_t", None)
                    none_elapsed = (now - none_start) if none_start is not None else 0.0
                    if none_elapsed < self.none_hold_sec:
                        pass
                    else:
                        if tr["raw_cnt"] >= self.confirm_frames and tr.get("lock_cnt", 0) <= 0:
                            stable_group = "NONE"
                            stable_state = "NONE"

            tr["stable_state"] = stable_state
            tr["stable_group"] = stable_group

            state = stable_state

            # Marker
            m = Marker()
            m.header.frame_id = "camera_depth_optical_frame"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "tracked"
            m.id = tid
            m.type = Marker.CUBE
            m.action = Marker.ADD

            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(z_now)
            m.pose.orientation.w = 1.0

            m.scale.x = m.scale.y = 0.4
            m.scale.z = 1.0

            m.lifetime.sec = 0
            m.lifetime.nanosec = int(0.3 * 1e9)

            if state == "STRONG_APPROACH":
                m.color.r, m.color.a = 1.0, 0.9
            elif state == "WEAK_APPROACH":
                m.color.r, m.color.g, m.color.a = 1.0, 1.0, 0.7
            elif state == "CROSSING":
                m.color.b, m.color.a = 1.0, 0.7
            else:
                m.color.g, m.color.a = 1.0, 0.5

            out.markers.append(m)

            # Debug
            z_raw = tr.get("z_last", float("nan"))
            none_start = tr.get("none_start_t", None)
            none_elapsed = (now - none_start) if none_start is not None else 0.0

            debug_lines.append(
                f"{tid}:{state}"
                f",raw={raw_state}"
                f",grp={stable_group}"
                f",rgb={rgb_match}"
                f",pos=({pos[0]:.2f},{pos[1]:.2f},{z_now:.2f})"
                f",vxy=({vel[0]:.2f},{vel[1]:.2f})"
                f",vz={vz:.2f},fwd={forward:.2f},lat={lateral:.2f},dt={dt:.2f}"
                f",zraw={z_raw if np.isfinite(z_raw) else 'nan'}"
                f",zup={tr.get('z_updated', False)}"
                f",wcnt={tr.get('weak_cnt',0)},scnt={tr.get('strong_cnt',0)},mon={tr.get('z_mon_cnt',0)}"
                f",rawcnt={tr.get('raw_cnt',0)},lock={tr.get('lock_cnt',0)}"
                f",none_t={none_elapsed:.2f}"
            )

        self.pub.publish(out)
        self.pub_dbg.publish(String(data=" | ".join(debug_lines)))


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
