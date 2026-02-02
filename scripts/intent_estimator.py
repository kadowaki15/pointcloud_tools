#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String
from collections import deque
import numpy as np
import time
import math


def radial_tangential(x, z, vx, vz):
    """Return (v_rad, v_tan, r)
    v_rad: + means approaching camera (range decreasing)
    """
    r = math.hypot(x, z)
    if r < 1e-4:
        return 0.0, 0.0, r
    # range rate dr/dt = (x*vx + z*vz)/r
    dr = (x * vx + z * vz) / r
    v_rad = -dr  # + => approaching
    v_tan_sq = max(0.0, (vx * vx + vz * vz) - (dr * dr))
    v_tan = math.sqrt(v_tan_sq)
    return float(v_rad), float(v_tan), float(r)


class IntentEstimator(Node):
    """
    IntentEstimator (ROS2 Humble)
    - Input : /tracked_markers (MarkerArray, ns='tracked')
    - Output: /object_intents (String) "id:state:vr:vt:vx:vz:r:z"
    - Viz   : /intent_markers TEXT, "ID{tid}:{state}"  (no TTC)

    改善点（横切り→approach_fast誤判定を減らす）:
    - approach_fast を許可するのは「真正面寄り」のときだけ
      1) vt が小さい（横移動が小さい）
      2) vr が vt より十分大きい（接近が支配的）
    - z<=0.05 など「前方距離として壊れている測定」は reject
    """

    def __init__(self):
        super().__init__('intent_estimator')

        # topics
        self.declare_parameter('input_topic', '/tracked_markers')
        self.declare_parameter('out_topic', '/object_intents')
        self.declare_parameter('viz_topic', '/intent_markers')
        self.declare_parameter('history_len', 8)

        # thresholds (range-based)
        self.declare_parameter('v_static_thresh', 0.08)      # m/s below => static
        self.declare_parameter('vr_thresh', 0.05)            # m/s radial > => approach
        self.declare_parameter('vt_thresh', 0.12)            # m/s tangential > => crossing

        # sudden approach (range-based)
        self.declare_parameter('vr_fast_thresh', 0.22)       # m/s
        self.declare_parameter('vr_accel_thresh', 0.80)      # m/s^2
        self.declare_parameter('sudden_confirm_frames', 2)
        self.declare_parameter('sudden_hold_sec', 0.6)

        # ★ fast を「真正面寄り」に限定するゲート（ここが今回の本命）
        # 横切りで少し近づくと vr が立つが、vt が大きいなら fast にしない
        self.declare_parameter('fast_vt_max', 0.20)          # vt がこれ超えたら fast 禁止
        # vr / (vt+eps) が小さい = 横成分が強い = fast 禁止
        self.declare_parameter('fast_dir_ratio_min', 3.0)    # まず 3.0 推奨（強め）
        # fast を見る距離上限（遠距離の揺れで fast が出るのを抑える）
        self.declare_parameter('fast_r_max', 2.0)            # まず 2.0m

        # hysteresis
        self.declare_parameter('state_confirm_frames', 4)
        self.declare_parameter('track_timeout', 2.0)

        # hold approach family to prevent flicker
        self.declare_parameter('approach_hold_sec', 0.8)
        self.declare_parameter('leave_approach_vr_away_thresh', 0.02)
        self.declare_parameter('leave_approach_extra_frames', 2)

        # viz
        self.declare_parameter('text_scale_z', 0.18)
        self.declare_parameter('text_lifetime_sec', 1.0)
        self.declare_parameter('publish_empty', True)
        self.declare_parameter('suppress_passing_by_output', True)

        # reject
        self.declare_parameter('reject_origin_xz', True)
        self.declare_parameter('origin_xz_eps', 0.03)
        self.declare_parameter('reject_ghost_center', True)
        self.declare_parameter('ghost_center_x', 0.0)
        self.declare_parameter('ghost_center_z', 0.5)
        self.declare_parameter('ghost_eps_x', 0.05)
        self.declare_parameter('ghost_eps_z', 0.06)

        # ★追加：前方距離として壊れてる測定を捨てる（z<=0 が出ると後段が死ぬ）
        self.declare_parameter('reject_nonpositive_z', True)
        self.declare_parameter('min_valid_z', 0.05)          # 5cm未満は無効扱い

        # read
        self.in_topic = str(self.get_parameter('input_topic').value)
        self.out_topic = str(self.get_parameter('out_topic').value)
        self.viz_topic = str(self.get_parameter('viz_topic').value)
        self.history_len = int(self.get_parameter('history_len').value)

        self.v_static_thresh = float(self.get_parameter('v_static_thresh').value)
        self.vr_thresh = float(self.get_parameter('vr_thresh').value)
        self.vt_thresh = float(self.get_parameter('vt_thresh').value)

        self.vr_fast_thresh = float(self.get_parameter('vr_fast_thresh').value)
        self.vr_accel_thresh = float(self.get_parameter('vr_accel_thresh').value)
        self.sudden_confirm_frames = int(self.get_parameter('sudden_confirm_frames').value)
        self.sudden_hold_sec = float(self.get_parameter('sudden_hold_sec').value)

        self.fast_vt_max = float(self.get_parameter('fast_vt_max').value)
        self.fast_dir_ratio_min = float(self.get_parameter('fast_dir_ratio_min').value)
        self.fast_r_max = float(self.get_parameter('fast_r_max').value)

        self.state_confirm_frames = int(self.get_parameter('state_confirm_frames').value)
        self.track_timeout = float(self.get_parameter('track_timeout').value)

        self.approach_hold_sec = float(self.get_parameter('approach_hold_sec').value)
        self.leave_approach_vr_away_thresh = float(self.get_parameter('leave_approach_vr_away_thresh').value)
        self.leave_approach_extra_frames = int(self.get_parameter('leave_approach_extra_frames').value)

        self.text_scale_z = float(self.get_parameter('text_scale_z').value)
        self.text_lifetime_sec = float(self.get_parameter('text_lifetime_sec').value)
        self.publish_empty = bool(self.get_parameter('publish_empty').value)
        self.suppress_passing_by_output = bool(self.get_parameter('suppress_passing_by_output').value)

        self.reject_origin_xz = bool(self.get_parameter('reject_origin_xz').value)
        self.origin_xz_eps = float(self.get_parameter('origin_xz_eps').value)
        self.reject_ghost_center = bool(self.get_parameter('reject_ghost_center').value)
        self.ghost_center_x = float(self.get_parameter('ghost_center_x').value)
        self.ghost_center_z = float(self.get_parameter('ghost_center_z').value)
        self.ghost_eps_x = float(self.get_parameter('ghost_eps_x').value)
        self.ghost_eps_z = float(self.get_parameter('ghost_eps_z').value)

        self.reject_nonpositive_z = bool(self.get_parameter('reject_nonpositive_z').value)
        self.min_valid_z = float(self.get_parameter('min_valid_z').value)

        # ros
        self.sub = self.create_subscription(MarkerArray, self.in_topic, self.cb_markers, 10)
        self.pub = self.create_publisher(String, self.out_topic, 10)
        self.pub_viz = self.create_publisher(MarkerArray, self.viz_topic, 10)

        # tracks
        self.tracks = {}

        self.get_logger().info(
            "IntentEstimator started (range-based vr/vt). "
            f"fast gate: vt<{self.fast_vt_max}, vr/vt>{self.fast_dir_ratio_min}, r<{self.fast_r_max}"
        )

    @staticmethod
    def _is_approach_family(state: str) -> bool:
        return state.startswith('approach')

    def _reject_point(self, x: float, z: float) -> bool:
        if self.reject_nonpositive_z and (z <= self.min_valid_z):
            return True
        if self.reject_origin_xz and (abs(x) < self.origin_xz_eps and abs(z) < self.origin_xz_eps):
            return True
        if self.reject_ghost_center:
            if (abs(x - self.ghost_center_x) < self.ghost_eps_x) and (abs(z - self.ghost_center_z) < self.ghost_eps_z):
                return True
        return False

    def _make_delete_marker(self, frame_id: str, stamp, tid: int) -> Marker:
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = 'intent'
        m.id = int(tid)
        m.action = Marker.DELETE
        return m

    def _allow_fast_gate(self, vr: float, vt: float, r: float) -> bool:
        """approach_fast を許可するか（横切り誤判定を減らすゲート）"""
        if r <= 1e-3:
            return False
        if r > float(self.fast_r_max):
            return False
        if vt > float(self.fast_vt_max):
            return False
        ratio = float(vr) / max(float(vt), 1e-3)
        if ratio < float(self.fast_dir_ratio_min):
            return False
        return True

    def cb_markers(self, msg: MarkerArray):
        tnow = time.time()

        if len(msg.markers) > 0:
            frame_id = msg.markers[0].header.frame_id or 'camera_depth_optical_frame'
            stamp = msg.markers[0].header.stamp
        else:
            frame_id = 'camera_depth_optical_frame'
            stamp = self.get_clock().now().to_msg()

        # ingest
        for m in msg.markers:
            if m.ns != 'tracked':
                continue
            if int(m.action) != int(Marker.ADD):
                continue

            tid = int(m.id)
            x = float(m.pose.position.x)
            z = float(m.pose.position.z)

            if self._reject_point(x, z):
                continue

            if tid not in self.tracks:
                self.tracks[tid] = {
                    'hist': deque(maxlen=self.history_len),
                    'last_seen': tnow,
                    'state': 'unknown',
                    'cand': 'unknown',
                    'cand_cnt': 0,
                    'hold_until': 0.0,

                    # for sudden
                    'last_vr': 0.0,
                    'last_vr_t': tnow,
                    'sudden_cand_cnt': 0,
                    'sudden_hold_until': 0.0,
                }

            self.tracks[tid]['hist'].append((tnow, x, z))
            self.tracks[tid]['last_seen'] = tnow

        # stale remove
        stale_ids = [tid for tid, v in self.tracks.items() if (tnow - v['last_seen']) > self.track_timeout]
        for tid in stale_ids:
            del self.tracks[tid]

        viz = MarkerArray()
        for tid in stale_ids:
            viz.markers.append(self._make_delete_marker(frame_id, stamp, tid))

        # evaluate
        for tid, info in self.tracks.items():
            hist = list(info['hist'])
            if len(hist) < 2:
                continue

            ts = np.array([h[0] for h in hist], dtype=np.float64)
            xs = np.array([h[1] for h in hist], dtype=np.float32)
            zs = np.array([h[2] for h in hist], dtype=np.float32)

            dt = float(ts[-1] - ts[0])
            if dt < 1e-4:
                continue

            vx = float((xs[-1] - xs[0]) / dt)
            vz = float((zs[-1] - zs[0]) / dt)

            x_now = float(xs[-1])
            z_now = float(zs[-1])

            # 念のため（ここで z が壊れてたら publish しない）
            if self.reject_nonpositive_z and (z_now <= self.min_valid_z):
                continue

            speed = float(np.hypot(vx, vz))
            vr, vt, r = radial_tangential(x_now, z_now, vx, vz)  # vr>0 approaching

            # vr accel (for sudden)
            vr_prev = float(info.get('last_vr', 0.0))
            t_prev = float(info.get('last_vr_t', tnow))
            dt_v = max(1e-3, float(tnow - t_prev))
            vr_accel = (vr - vr_prev) / dt_v
            info['last_vr'] = vr
            info['last_vr_t'] = tnow

            # base candidate
            if speed < self.v_static_thresh:
                candidate = 'static'
            else:
                if vr < self.vr_thresh and vt > self.vt_thresh:
                    candidate = 'crossing'
                elif vr > self.vr_thresh:
                    candidate = 'approach'
                else:
                    candidate = 'passing_by'

            # sudden overlay (approach_fast)
            sudden_now = (vr > self.vr_thresh) and ((vr > self.vr_fast_thresh) or (vr_accel > self.vr_accel_thresh))

            # ★追加ゲート：横切りっぽいときは fast を殺す
            if sudden_now:
                if not self._allow_fast_gate(vr, vt, r):
                    sudden_now = False

            if sudden_now:
                info['sudden_cand_cnt'] = info.get('sudden_cand_cnt', 0) + 1
            else:
                info['sudden_cand_cnt'] = 0

            if info['sudden_cand_cnt'] >= self.sudden_confirm_frames:
                info['sudden_hold_until'] = max(info.get('sudden_hold_until', 0.0), tnow + self.sudden_hold_sec)

            # sudden hold 中でも、fastゲートを通る時だけ fast にする（横切り誤爆の再発防止）
            if (tnow < info.get('sudden_hold_until', 0.0)) and candidate.startswith('approach'):
                if self._allow_fast_gate(vr, vt, r):
                    candidate = 'approach_fast'

            # hold approach family
            prev = info.get('state', 'unknown')
            if self._is_approach_family(candidate):
                info['hold_until'] = max(info.get('hold_until', 0.0), tnow + self.approach_hold_sec)

            if self._is_approach_family(prev) and (tnow < info.get('hold_until', 0.0)):
                if not self._is_approach_family(candidate):
                    candidate = prev

            # stricter leaving approach
            extra_need = 0
            if self._is_approach_family(prev) and (not self._is_approach_family(candidate)):
                if vr > self.leave_approach_vr_away_thresh:
                    candidate = prev
                else:
                    extra_need = self.leave_approach_extra_frames

            # hysteresis
            if candidate == info.get('cand', 'unknown'):
                info['cand_cnt'] = info.get('cand_cnt', 0) + 1
            else:
                info['cand'] = candidate
                info['cand_cnt'] = 1

            need = int(self.state_confirm_frames + extra_need)
            if candidate != prev and info['cand_cnt'] >= need:
                info['state'] = candidate
                info['cand'] = candidate
                info['cand_cnt'] = 0

            committed = info.get('state', 'unknown')
            publish_state = committed if committed != 'unknown' else candidate

            if self.suppress_passing_by_output and publish_state == 'passing_by':
                publish_state = 'unknown'

            # publish string
            self.pub.publish(String(
                data=f"{tid}:{publish_state}:{vr:.3f}:{vt:.3f}:{vx:.3f}:{vz:.3f}:{r:.3f}:{z_now:.3f}"
            ))

            # viz
            if publish_state in ('unknown', 'passing_by'):
                continue

            mm = Marker()
            mm.header.frame_id = frame_id
            mm.header.stamp = stamp
            mm.ns = 'intent'
            mm.id = int(tid)
            mm.type = Marker.TEXT_VIEW_FACING
            mm.action = Marker.ADD

            mm.pose.position.x = float(x_now)
            mm.pose.position.y = 0.0
            mm.pose.position.z = float(z_now)
            mm.scale.z = float(self.text_scale_z)

            if publish_state == 'approach_fast':
                mm.color.r, mm.color.g, mm.color.b, mm.color.a = 1.0, 0.0, 0.0, 1.0
            elif publish_state.startswith('approach'):
                mm.color.r, mm.color.g, mm.color.b, mm.color.a = 1.0, 0.6, 0.0, 1.0
            elif publish_state == 'crossing':
                mm.color.r, mm.color.g, mm.color.b, mm.color.a = 0.0, 0.7, 1.0, 1.0
            elif publish_state == 'static':
                mm.color.r, mm.color.g, mm.color.b, mm.color.a = 0.0, 1.0, 0.0, 1.0

            mm.text = f"ID{tid}:{publish_state}"

            sec = int(self.text_lifetime_sec)
            nsec = int((self.text_lifetime_sec - sec) * 1e9)
            mm.lifetime.sec = max(0, sec)
            mm.lifetime.nanosec = max(0, nsec)

            viz.markers.append(mm)

        if self.publish_empty or (len(viz.markers) > 0):
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
