#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
import time


def stamp_is_valid(stamp) -> bool:
    try:
        return (stamp.sec != 0) or (stamp.nanosec != 0)
    except Exception:
        return False


def create_pc2_xyz(points_xyz: np.ndarray, frame_id: str, stamp=None):
    header = Header()
    header.frame_id = frame_id
    if stamp is None or not stamp_is_valid(stamp):
        header.stamp = rclpy.time.Time(seconds=int(time.time())).to_msg()
    else:
        header.stamp = stamp
    return pc2.create_cloud_xyz32(header, points_xyz.tolist())


class RGBMotionGateFilter(Node):
    """
    RGBフレーム差分 -> motion mask を生成し、
    /pointcloud_diff_comp_raw を mask でゲートして /pointcloud_diff_comp を publish する。

    ✅ この完成版の方針（パイプラインを殺さない）:
    - デフォルトtopicは /camera/camera/... に固定（あなたのtopic list準拠）
    - mask/camera_info未取得でも /pointcloud_diff_comp を "rawでパススルー" できる
    - /rgb_motion_mask は真っ黒でも毎フレーム publish
    - ★落としすぎ救済：filtered が少なすぎるフレームは raw を通す（approach/crossing を殺さない）
    - keep率ログで「落としすぎ」を即判定
    """

    def __init__(self):
        super().__init__('rgb_motion_gate_filter')

        # -------------------------
        # Topics (あなたの環境に合わせてデフォルト固定)
        # -------------------------
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        # 重要：点群が depth 系なら /camera/camera/depth/camera_info の方が合う場合が多い
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('pc_in_topic', '/pointcloud_diff_comp_raw')
        self.declare_parameter('mask_out_topic', '/rgb_motion_mask')
        self.declare_parameter('pc_out_topic', '/pointcloud_diff_comp')

        # -------------------------
        # Fail-safe
        # -------------------------
        self.declare_parameter('passthrough_if_no_mask', True)

        # -------------------------
        # Motion detection (出やすい寄り)
        # -------------------------
        self.declare_parameter('diff_thresh', 32)
        self.declare_parameter('min_area', 700)
        self.declare_parameter('median_ksize', 5)      # odd >=3
        self.declare_parameter('morph_ksize', 3)       # >=3
        self.declare_parameter('open_iter', 1)
        self.declare_parameter('close_iter', 1)

        # -------------------------
        # Mask stability / tolerance
        # -------------------------
        self.declare_parameter('mask_dilate_ksize', 2)   # ズレ吸収（上げすぎると誤検知も増える）
        self.declare_parameter('mask_hold_frames', 3)    # OR masks for N frames
        self.declare_parameter('mask_min_value', 1)

        # -------------------------
        # Point filtering
        # -------------------------
        self.declare_parameter('min_z', 0.15)
        self.declare_parameter('max_z', 5.0)
        self.declare_parameter('max_points', 60000)

        # -------------------------
        # ★落としすぎ救済（ここが今回の主役）
        # -------------------------
        # 例）入力の 1% 未満しか残らない or 80点未満なら「落としすぎ」とみなして raw を流す
        self.declare_parameter('min_keep_ratio', 0.01)   # 0.01 = 1%
        self.declare_parameter('min_keep_points', 80)

        # -------------------------
        # Debug
        # -------------------------
        self.declare_parameter('log_every_n_pc', 15)  # 何callbackに1回ログを出すか

        # Read params
        self.color_topic = str(self.get_parameter('color_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.pc_in_topic = str(self.get_parameter('pc_in_topic').value)
        self.mask_out_topic = str(self.get_parameter('mask_out_topic').value)
        self.pc_out_topic = str(self.get_parameter('pc_out_topic').value)

        self.passthrough_if_no_mask = bool(self.get_parameter('passthrough_if_no_mask').value)

        self.diff_thresh = int(self.get_parameter('diff_thresh').value)
        self.min_area = int(self.get_parameter('min_area').value)
        self.median_ksize = int(self.get_parameter('median_ksize').value)
        self.morph_ksize = int(self.get_parameter('morph_ksize').value)
        self.open_iter = int(self.get_parameter('open_iter').value)
        self.close_iter = int(self.get_parameter('close_iter').value)

        self.mask_dilate_ksize = int(self.get_parameter('mask_dilate_ksize').value)
        self.mask_hold_frames = int(self.get_parameter('mask_hold_frames').value)
        self.mask_min_value = int(self.get_parameter('mask_min_value').value)

        self.min_z = float(self.get_parameter('min_z').value)
        self.max_z = float(self.get_parameter('max_z').value)
        self.max_points = int(self.get_parameter('max_points').value)

        self.min_keep_ratio = float(self.get_parameter('min_keep_ratio').value)
        self.min_keep_points = int(self.get_parameter('min_keep_points').value)

        self.log_every_n_pc = int(self.get_parameter('log_every_n_pc').value)
        self._pc_cb_count = 0

        # Intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.bridge = CvBridge()
        self.prev_gray = None

        # Latest mask (held OR buffer)
        self.mask_hold = []
        self.latest_mask = None
        self.latest_mask_header = None

        # Sub/Pub
        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.cb_info, 10)
        self.sub_color = self.create_subscription(Image, self.color_topic, self.cb_color, 10)
        self.sub_pc = self.create_subscription(PointCloud2, self.pc_in_topic, self.cb_pc, 10)

        self.pub_mask = self.create_publisher(Image, self.mask_out_topic, 10)
        self.pub_pc = self.create_publisher(PointCloud2, self.pc_out_topic, 10)

        self.get_logger().info("RGBMotionGateFilter FINAL+SAFE started")
        self.get_logger().info(f"color_topic       : {self.color_topic}")
        self.get_logger().info(f"camera_info_topic : {self.camera_info_topic}")
        self.get_logger().info(f"pc_in_topic       : {self.pc_in_topic}")
        self.get_logger().info(f"mask_out_topic    : {self.mask_out_topic}")
        self.get_logger().info(f"pc_out_topic      : {self.pc_out_topic}")
        self.get_logger().info(f"passthrough_if_no_mask: {self.passthrough_if_no_mask}")
        self.get_logger().info(
            f"defaults: diff_thresh={self.diff_thresh} min_area={self.min_area} "
            f"morph={self.morph_ksize} hold={self.mask_hold_frames} dilate={self.mask_dilate_ksize}"
        )
        self.get_logger().info(
            f"failsafe: min_keep_ratio={self.min_keep_ratio} min_keep_points={self.min_keep_points}"
        )

    # -------------------------
    # CameraInfo
    # -------------------------
    def cb_info(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    # -------------------------
    # Color -> motion mask
    # -------------------------
    def cb_color(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            motion = np.zeros_like(gray, dtype=np.uint8)
        else:
            diff = cv2.absdiff(self.prev_gray, gray)
            _, motion = cv2.threshold(diff, self.diff_thresh, 255, cv2.THRESH_BINARY)

            if self.median_ksize >= 3 and self.median_ksize % 2 == 1:
                motion = cv2.medianBlur(motion, self.median_ksize)

            if self.morph_ksize >= 3:
                k = self.morph_ksize
                kernel = np.ones((k, k), np.uint8)
                if self.open_iter > 0:
                    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, iterations=self.open_iter)
                if self.close_iter > 0:
                    motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel, iterations=self.close_iter)

            contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean = np.zeros_like(motion)
            for c in contours:
                if cv2.contourArea(c) >= self.min_area:
                    cv2.drawContours(clean, [c], -1, 255, thickness=-1)
            motion = clean

            self.prev_gray = gray

        # dilate（投影ズレ吸収）
        if self.mask_dilate_ksize and self.mask_dilate_ksize > 1:
            k = self.mask_dilate_ksize
            kernel = np.ones((k, k), np.uint8)
            motion = cv2.dilate(motion, kernel, iterations=1)

        # hold frames (OR)
        self.mask_hold.append(motion)
        if len(self.mask_hold) > max(1, self.mask_hold_frames):
            self.mask_hold = self.mask_hold[-self.mask_hold_frames:]

        merged = self.mask_hold[0].copy()
        for m in self.mask_hold[1:]:
            if m.shape == merged.shape:
                merged = cv2.bitwise_or(merged, m)
            else:
                self.mask_hold = [motion]
                merged = motion.copy()
                break

        self.latest_mask = merged
        self.latest_mask_header = msg.header

        # 真っ黒でも publish（重要）
        mask_msg = self.bridge.cv2_to_imgmsg(merged, encoding='mono8')
        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)

    # -------------------------
    # PointCloud2 utils
    # -------------------------
    def _pc2_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        pts = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if len(pts) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)

    # -------------------------
    # Gate pointcloud by mask
    # -------------------------
    def cb_pc(self, msg: PointCloud2):
        self._pc_cb_count += 1

        pts = self._pc2_to_xyz(msg)
        if pts.shape[0] == 0:
            self.pub_pc.publish(create_pc2_xyz(np.empty((0, 3), np.float32), msg.header.frame_id, msg.header.stamp))
            return

        # mask or intrinsics 未取得 → フェイルセーフ
        if self.latest_mask is None or self.fx is None:
            if self.passthrough_if_no_mask:
                self.pub_pc.publish(create_pc2_xyz(pts.astype(np.float32), msg.header.frame_id, msg.header.stamp))
            return

        # cap points for speed
        if self.max_points is not None and pts.shape[0] > self.max_points:
            d = np.linalg.norm(pts, axis=1)
            idx = np.argsort(d)[:self.max_points]
            pts = pts[idx]

        h, w = self.latest_mask.shape[:2]

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        valid_z = (z > self.min_z) & (z < self.max_z)
        if not np.any(valid_z):
            self.pub_pc.publish(create_pc2_xyz(np.empty((0, 3), np.float32), msg.header.frame_id, msg.header.stamp))
            return

        pts_v = pts[valid_z]
        x = x[valid_z]
        y = y[valid_z]
        z = z[valid_z]

        # pinhole projection
        u = (self.fx * (x / z) + self.cx).astype(np.int32)
        v = (self.fy * (y / z) + self.cy).astype(np.int32)

        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(in_img):
            # 投影が合ってない可能性が高い：ここでもraw救済してOK
            self.pub_pc.publish(create_pc2_xyz(pts.astype(np.float32), msg.header.frame_id, msg.header.stamp))
            return

        pts_v = pts_v[in_img]
        u = u[in_img]
        v = v[in_img]

        keep = self.latest_mask[v, u] >= self.mask_min_value
        filtered = pts_v[keep]

        # -------------------------
        # ★落としすぎ救済（approach/crossingを殺さない）
        # -------------------------
        min_by_ratio = int(max(1, pts.shape[0] * self.min_keep_ratio))
        min_required = max(self.min_keep_points, min_by_ratio)

        used_raw_fallback = False
        if filtered.shape[0] < min_required:
            # ゲートがズレてる/弱すぎる/タイミングずれ → rawを通す
            out_pts = pts
            used_raw_fallback = True
        else:
            out_pts = filtered

        self.pub_pc.publish(create_pc2_xyz(out_pts.astype(np.float32), msg.header.frame_id, msg.header.stamp))

        # --- debug log (keep率 + fallback) ---
        if self.log_every_n_pc > 0 and (self._pc_cb_count % self.log_every_n_pc == 0):
            keep_ratio = 100.0 * (filtered.shape[0] / max(1, pts.shape[0]))
            self.get_logger().info(
                f"pc gate: in={pts.shape[0]} -> kept={filtered.shape[0]} ({keep_ratio:.1f}%) "
                f"fallback={'RAW' if used_raw_fallback else 'GATED'} "
                f"(min_required={min_required}) "
                f"diff_thr={self.diff_thresh} min_area={self.min_area} hold={self.mask_hold_frames}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = RGBMotionGateFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
