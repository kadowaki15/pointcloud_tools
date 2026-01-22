#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String
import numpy as np
from collections import deque
import time

# =================================================
# Simple Kalman Filter (x, y, vx, vy)
# =================================================
class SimpleKalman:
    def __init__(self, dt=0.1, q=1e-2, r=1e-1):
        self.dt = dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float32)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.Q = np.diag([q, q, q*10, q*10])
        self.R = np.diag([r, r])
        self.P = np.eye(4)
        self.x = np.zeros(4, dtype=np.float32)

    def init(self, x, y):
        self.x[:] = [x, y, 0.0, 0.0]
        self.P = np.eye(4) * 0.5

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, zx, zy):
        z = np.array([zx, zy], dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def pos(self):
        return self.x[:2]

    def vel(self):
        return self.x[2:]


# =================================================
# Tracker Node
# =================================================
class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_with_rgb_and_states')

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter('input_topic', '/dynamic_markers')
        self.declare_parameter('rgb_topic', '/rgb_detections_markers')
        self.declare_parameter('output_topic', '/tracked_markers')
        self.declare_parameter('debug_topic', '/tracked_info')

        self.declare_parameter('match_dist', 0.8)
        self.declare_parameter('ttl', 15)
        self.declare_parameter('history_len', 10)

        self.declare_parameter('weak_approach_thresh', 0.04)
        self.declare_parameter('strong_approach_thresh', 0.08)
        self.declare_parameter('crossing_thresh', 0.05)

        self.declare_parameter('rgb_match_dist', 0.6)

        # -------------------------
        # Load parameters
        # -------------------------
        self.input_topic = self.get_parameter('input_topic').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.debug_topic = self.get_parameter('debug_topic').value

        self.match_dist = self.get_parameter('match_dist').value
        self.ttl = self.get_parameter('ttl').value
        self.history_len = self.get_parameter('history_len').value

        self.weak_th = self.get_parameter('weak_approach_thresh').value
        self.strong_th = self.get_parameter('strong_approach_thresh').value
        self.cross_th = self.get_parameter('crossing_thresh').value

        self.rgb_match_dist = self.get_parameter('rgb_match_dist').value

        # -------------------------
        # ROS I/O
        # -------------------------
        self.sub = self.create_subscription(
            MarkerArray, self.input_topic, self.cb, 10)

        self.sub_rgb = self.create_subscription(
            MarkerArray, self.rgb_topic, self.rgb_cb, 10)

        self.pub = self.create_publisher(
            MarkerArray, self.output_topic, 10)

        self.pub_dbg = self.create_publisher(
            String, self.debug_topic, 10)

        # -------------------------
        # Internal state
        # -------------------------
        self.tracks = {}
        self.next_id = 0
        self.frame = 0
        self.latest_rgb = []

        self.get_logger().info("Tracker initialized (approach 2-stage + crossing + RGB)")

    # -------------------------------------------------
    # RGB callback
    # -------------------------------------------------
    def rgb_cb(self, msg: MarkerArray):
        self.latest_rgb = []
        for m in msg.markers:
            self.latest_rgb.append(
                np.array([m.pose.position.x, m.pose.position.y], dtype=np.float32)
            )

    # -------------------------------------------------
    # Main callback
    # -------------------------------------------------
    def cb(self, msg: MarkerArray):
        self.frame += 1
        detections = [
            np.array([m.pose.position.x, m.pose.position.y], dtype=np.float32)
            for m in msg.markers
        ]

        out = MarkerArray()
        debug_lines = []

        # ---------- Predict ----------
        for tr in self.tracks.values():
            tr['kf'].predict()
            tr['history'].append(tr['kf'].pos())
            if len(tr['history']) > self.history_len:
                tr['history'].popleft()

        # ---------- Associate ----------
        for det in detections:
            best_id, best_d = None, float('inf')
            for tid, tr in self.tracks.items():
                d = np.linalg.norm(det - tr['kf'].pos())
                if d < best_d and d < self.match_dist:
                    best_d, best_id = d, tid

            if best_id is not None:
                self.tracks[best_id]['kf'].update(det[0], det[1])
                self.tracks[best_id]['last'] = self.frame
            else:
                kf = SimpleKalman()
                kf.init(det[0], det[1])
                self.tracks[self.next_id] = {
                    'kf': kf,
                    'history': deque([det], maxlen=self.history_len),
                    'last': self.frame,
                    'weak_cnt': 0,
                    'strong_cnt': 0
                }
                self.next_id += 1

        # ---------- Remove old (DELETE Marker) ----------
        for tid in list(self.tracks.keys()):
            if self.frame - self.tracks[tid]['last'] > self.ttl:
                m = Marker()
                m.header.frame_id = 'camera_depth_optical_frame'
                m.ns = 'tracked'
                m.id = tid
                m.action = Marker.DELETE
                out.markers.append(m)
                del self.tracks[tid]

        # ---------- State estimation & publish ----------
        for tid, tr in self.tracks.items():
            vel = tr['kf'].vel()
            pos = tr['kf'].pos()

            speed = np.linalg.norm(vel)
            forward = -vel[0]          # 仮：x負方向が接近
            lateral = abs(vel[1])

            # --- RGB match ---
            rgb_match = any(
                np.linalg.norm(rgb - pos) < self.rgb_match_dist
                for rgb in self.latest_rgb
            )

            # --- State ---
            state = "NONE"

            if forward > self.weak_th:
                tr['weak_cnt'] += 1
            else:
                tr['weak_cnt'] = 0

            if forward > self.strong_th:
                tr['strong_cnt'] += 1
            else:
                tr['strong_cnt'] = 0

            if tr['strong_cnt'] >= (2 if rgb_match else 3):
                state = "STRONG_APPROACH"
            elif tr['weak_cnt'] >= 2:
                state = "WEAK_APPROACH"
            elif lateral > self.cross_th and abs(forward) < self.weak_th:
                state = "CROSSING"

            # --- Marker ---
            m = Marker()
            m.header.frame_id = 'camera_depth_optical_frame'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'tracked'
            m.id = tid
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.4
            m.scale.z = 1.0

            # lifetime（重要：残らない）
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
            debug_lines.append(f"{tid}:{state},rgb={rgb_match}")

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


if __name__ == '__main__':
    main()
