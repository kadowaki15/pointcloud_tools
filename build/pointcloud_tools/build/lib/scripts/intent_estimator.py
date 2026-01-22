#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String
from collections import deque
import numpy as np
import time
import math

def radial_tangential(p, v):
    """Return (v_rad, v_tan, range).
    v_rad: positive => moving TOWARD camera (we invert dot sign to make this true).
    """
    r = np.linalg.norm(p)
    if r < 1e-4:
        return 0.0, 0.0, r
    u = p / r
    # make v_rad positive when velocity has component toward camera
    v_rad = -float(np.dot(v, u))
    v_tan_vec = v - np.dot(v, u) * u
    v_tan = float(np.linalg.norm(v_tan_vec))
    return v_rad, v_tan, r

class IntentEstimator(Node):
    def __init__(self):
        super().__init__('intent_estimator')
        # topics / basic params
        self.declare_parameter('input_topic', '/tracked_markers')
        self.declare_parameter('out_topic', '/object_intents')
        self.declare_parameter('viz_topic', '/intent_markers')
        self.declare_parameter('history_len', 6)

        # tuned thresholds
        self.declare_parameter('v_static_thresh', 0.08)   # below = static
        self.declare_parameter('v_rad_thresh', 0.05)      # minimal radial speed to consider
        self.declare_parameter('v_tan_thresh', 0.12)      # minimal tangential speed to consider crossing
        self.declare_parameter('ttc_warn', 2.2)
        self.declare_parameter('ttc_urgent', 1.0)

        # angle-based decision (degrees)
        self.declare_parameter('approach_angle_deg', 35.0)   # angle <= this => treat as approach-dominant
        self.declare_parameter('cross_angle_deg', 70.0)     # angle >= this => treat as crossing-dominant

        # hysteresis / confirmation
        self.declare_parameter('state_confirm_frames', 2)   # how many consecutive frames to switch
        self.declare_parameter('track_timeout', 2.0)

        # read params
        self.in_topic = str(self.get_parameter('input_topic').value)
        self.out_topic = str(self.get_parameter('out_topic').value)
        self.viz_topic = str(self.get_parameter('viz_topic').value)
        self.history_len = int(self.get_parameter('history_len').value)

        self.v_static_thresh = float(self.get_parameter('v_static_thresh').value)
        self.v_rad_thresh = float(self.get_parameter('v_rad_thresh').value)
        self.v_tan_thresh = float(self.get_parameter('v_tan_thresh').value)
        self.ttc_warn = float(self.get_parameter('ttc_warn').value)
        self.ttc_urgent = float(self.get_parameter('ttc_urgent').value)

        self.approach_angle_deg = float(self.get_parameter('approach_angle_deg').value)
        self.cross_angle_deg = float(self.get_parameter('cross_angle_deg').value)

        self.state_confirm_frames = int(self.get_parameter('state_confirm_frames').value)
        self.track_timeout = float(self.get_parameter('track_timeout').value)

        # ROS pubs/subs
        self.sub = self.create_subscription(MarkerArray, self.in_topic, self.cb_markers, 10)
        self.pub = self.create_publisher(String, self.out_topic, 10)
        self.pub_viz = self.create_publisher(MarkerArray, self.viz_topic, 10)

        # per-track storage: hist, last_seen, state, state_cnt (consecutive candidate count)
        self.tracks = {}  # tid -> {'hist':deque, 'last_seen':t, 'state':str, 'state_cnt':int}

        self.get_logger().info("IntentEstimator (angle + hysteresis) started")

    def _angle_deg(self, v_rad, v_tan):
        # angle between radial axis and velocity: 0 = directly radial (approach), 90 = purely tangential
        # guard v_rad>=0 for arctan2
        return math.degrees(math.atan2(v_tan, max(v_rad, 1e-9)))

    def cb_markers(self, msg: MarkerArray):
        tnow = time.time()
        # parse markers
        for m in msg.markers:
            if m.ns not in ('tracked', 'tracked_text'):
                continue
            if m.ns == 'tracked_text':
                continue
            tid = int(m.id)
            px = float(m.pose.position.x)
            py = float(m.pose.position.y)
            if tid not in self.tracks:
                self.tracks[tid] = {
                    'hist': deque(maxlen=self.history_len),
                    'last_seen': tnow,
                    'state': 'unknown',
                    'state_cnt': 0
                }
            self.tracks[tid]['hist'].append((tnow, px, py))
            self.tracks[tid]['last_seen'] = tnow

        # remove stale
        stale = [tid for tid, v in self.tracks.items() if (tnow - v['last_seen']) > self.track_timeout]
        for s in stale:
            del self.tracks[s]

        viz = MarkerArray()
        # evaluate each track
        for tid, info in self.tracks.items():
            hist = list(info['hist'])
            if len(hist) < 2:
                continue
            # simple linear velocity estimate over history
            ts = np.array([h[0] for h in hist])
            xs = np.array([h[1] for h in hist])
            ys = np.array([h[2] for h in hist])
            dt = ts[-1] - ts[0]
            if dt < 1e-4:
                continue
            vx = float((xs[-1] - xs[0]) / dt)
            vy = float((ys[-1] - ys[0]) / dt)
            p = np.array([xs[-1], ys[-1]])
            vvec = np.array([vx, vy])
            speed = float(np.linalg.norm(vvec))
            v_rad, v_tan, rng = radial_tangential(p, vvec)
            angle_deg = self._angle_deg(v_rad, v_tan)

            # candidate decision (angle + thresholds)
            candidate = 'unknown'
            ttc = float('nan')
            if speed < self.v_static_thresh:
                candidate = 'static'
            else:
                # if radial negligible and tangential large => crossing
                if v_rad < self.v_rad_thresh and v_tan > self.v_tan_thresh:
                    candidate = 'crossing'
                else:
                    # if radial points away (v_rad <= 0) -> passing_by
                    if v_rad <= 0.0:
                        candidate = 'passing_by'
                    else:
                        # radial positive: check angle
                        if angle_deg <= self.approach_angle_deg:
                            # approach family: evaluate TTC
                            ttc = rng / v_rad if v_rad > 1e-6 else float('inf')
                            if ttc < self.ttc_urgent:
                                candidate = 'approach_urgent'
                            elif ttc < self.ttc_warn:
                                candidate = 'approach_warn'
                            else:
                                candidate = 'approach'
                        elif angle_deg >= self.cross_angle_deg:
                            candidate = 'crossing'
                        else:
                            # mid-angle: prefer approach if radial sizeable, else crossing
                            if v_rad > self.v_rad_thresh:
                                ttc = rng / v_rad if v_rad > 1e-6 else float('inf')
                                if ttc < self.ttc_warn:
                                    candidate = 'approach_warn'
                                else:
                                    candidate = 'approach'
                            else:
                                candidate = 'crossing'

            # hysteresis: require consecutive candidate frames to switch
            prev = info.get('state', 'unknown')
            if candidate == prev:
                info['state_cnt'] = 0  # reset; already in state
            else:
                # increment counter (or set to 1 if first time)
                info['state_cnt'] = info.get('state_cnt', 0) + 1
                # require N frames to commit change
                need = self.state_confirm_frames
                # make leaving 'approach_urgent' harder (optional) — keep symmetric for simplicity
                if info['state_cnt'] >= need:
                    info['state'] = candidate
                    info['state_cnt'] = 0

            # publish intent using committed state
            committed = info['state']
            # if we haven't committed any state yet, use candidate for immediate feedback
            publish_state = committed if committed != 'unknown' else candidate

            out_ttc = ttc if not math.isnan(ttc) else float('nan')
            outstr = f"{tid}:{publish_state}:{out_ttc:.3f}:{v_rad:.3f}:{vx:.3f}:{vy:.3f}:{rng:.3f}"
            self.pub.publish(String(data=outstr))

            # visualization: show approach/crossing/static only (avoid passing_by/unknown)
            if publish_state in ('passing_by', 'unknown'):
                continue
            mm = Marker()
            mm.header.frame_id = msg.markers[0].header.frame_id if len(msg.markers) > 0 else 'camera_depth_optical_frame'
            mm.header.stamp = msg.markers[0].header.stamp if len(msg.markers) > 0 else self.get_clock().now().to_msg()
            mm.ns = 'intent'
            mm.id = int(tid)
            mm.type = Marker.TEXT_VIEW_FACING
            mm.action = Marker.ADD
            mm.pose.position.x = float(xs[-1])
            mm.pose.position.y = float(ys[-1])
            mm.pose.position.z = 2.0
            mm.scale.z = 0.22
            if publish_state.startswith('approach_urgent'):
                mm.color.r, mm.color.g, mm.color.b, mm.color.a = 1.0, 0.0, 0.0, 1.0
            elif publish_state.startswith('approach'):
                mm.color.r, mm.color.g, mm.color.b, mm.color.a = 1.0, 0.6, 0.0, 1.0
            elif publish_state == 'crossing':
                mm.color.r, mm.color.g, mm.color.b, mm.color.a = 0.0, 0.7, 1.0, 1.0
            elif publish_state == 'static':
                mm.color.r, mm.color.g, mm.color.b, mm.color.a = 0.0, 1.0, 0.0, 1.0
            mm.text = f"ID{tid}:{publish_state} a={angle_deg:.0f}"
            mm.lifetime.sec = 2
            mm.lifetime.nanosec = 0
            viz.markers.append(mm)

            # debug trace
            self.get_logger().debug(
                f"TID{tid} candidate={candidate} committed={info.get('state')} "
                f"v_rad={v_rad:.3f} v_tan={v_tan:.3f} angle={angle_deg:.1f} rng={rng:.2f} speed={speed:.3f}"
            )

        if len(viz.markers) > 0:
            self.pub_viz.publish(viz)

def main(args=None):
    rclpy.init(args=args)
    node = IntentEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
