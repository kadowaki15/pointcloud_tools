#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
from scipy.spatial import cKDTree
from std_msgs.msg import Header
import time

def transform_to_matrix(tf_stamped: TransformStamped):
    t = tf_stamped.transform.translation
    q = tf_stamped.transform.rotation
    tx, ty, tz = t.x, t.y, t.z
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n > 0:
        qx /= n; qy /= n; qz /= n; qw /= n

    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    rot = np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float32)

    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot
    mat[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return mat

def apply_transform(points: np.ndarray, mat: np.ndarray):
    if points.size == 0:
        return points
    hom = np.ones((points.shape[0], 4), dtype=np.float32)
    hom[:, :3] = points
    return (hom @ mat.T)[:, :3]

def stamp_is_valid(stamp) -> bool:
    try:
        return (stamp.sec != 0) or (stamp.nanosec != 0)
    except Exception:
        return False

def create_pointcloud2_from_xyz(points_xyz: np.ndarray, frame_id: str, stamp=None):
    header = Header()
    header.frame_id = frame_id
    if stamp is None or not stamp_is_valid(stamp):
        header.stamp = rclpy.time.Time(seconds=int(time.time())).to_msg()
    else:
        header.stamp = stamp
    return pc2.create_cloud_xyz32(header, points_xyz.tolist())

def voxel_downsample(points: np.ndarray, voxel: float, max_points: int = None) -> np.ndarray:
    if points.shape[0] == 0 or voxel <= 0.0:
        return points
    coords = np.floor(points / voxel).astype(np.int32)
    keys, inv = np.unique(coords, axis=0, return_inverse=True)

    out = np.zeros((keys.shape[0], 3), dtype=np.float32)
    counts = np.zeros((keys.shape[0],), dtype=np.int32)
    for i in range(points.shape[0]):
        k = inv[i]
        out[k] += points[i]
        counts[k] += 1
    out /= np.maximum(counts[:, None], 1)

    if max_points is not None and out.shape[0] > max_points:
        d = np.linalg.norm(out, axis=1)
        idx = np.argsort(d)[:max_points]
        out = out[idx]
    return out

class EgoCompensator(Node):
    def __init__(self):
        super().__init__('ego_compensator')

        self.declare_parameter('input_topic', '/camera/camera/depth/color/points')
        self.declare_parameter('out_topic', '/pointcloud_diff_comp_raw')

        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('max_points', 60000)

        self.declare_parameter('max_distance', 2.5)
        self.declare_parameter('min_distance', 0.2)

        self.declare_parameter('thr_base', 0.02)
        self.declare_parameter('thr_slope', 0.02)

        self.declare_parameter('ror_radius', 0.12)
        self.declare_parameter('ror_min_neighbors', 4)
        self.declare_parameter('min_dynamic_points', 30)

        self.declare_parameter('use_tf', True)
        self.declare_parameter('tf_timeout_sec', 0.02)

        self.in_topic = str(self.get_parameter('input_topic').value)
        self.out_topic = str(self.get_parameter('out_topic').value)

        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.max_points = int(self.get_parameter('max_points').value)

        self.max_distance = float(self.get_parameter('max_distance').value)
        self.min_distance = float(self.get_parameter('min_distance').value)

        self.thr_base = float(self.get_parameter('thr_base').value)
        self.thr_slope = float(self.get_parameter('thr_slope').value)

        self.ror_radius = float(self.get_parameter('ror_radius').value)
        self.ror_min_neighbors = int(self.get_parameter('ror_min_neighbors').value)
        self.min_dynamic_points = int(self.get_parameter('min_dynamic_points').value)

        self.use_tf = bool(self.get_parameter('use_tf').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.prev_points = None
        self.prev_frame_id = None
        self.prev_stamp = None

        self.sub = self.create_subscription(PointCloud2, self.in_topic, self.cb, 10)
        self.pub = self.create_publisher(PointCloud2, self.out_topic, 1)  # depth=1 推奨

        self.get_logger().info(f"EgoCompensator started: in={self.in_topic} out={self.out_topic}")

    def _pc2_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        pts = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if len(pts) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)

    def _range_filter(self, pts: np.ndarray) -> np.ndarray:
        if pts.shape[0] == 0:
            return pts
        d = np.linalg.norm(pts, axis=1)
        m = (d >= self.min_distance) & (d <= self.max_distance)
        return pts[m]

    def _try_get_tf_mat(self, target_frame: str, source_frame: str, stamp) -> np.ndarray:
        if not self.use_tf:
            return None
        try:
            # stamp指定 → ダメなら latest
            try:
                ts = rclpy.time.Time(seconds=stamp.sec, nanoseconds=stamp.nanosec)
                tf_stamped = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, ts,
                    timeout=Duration(seconds=self.tf_timeout_sec)
                )
                return transform_to_matrix(tf_stamped)
            except Exception:
                tf_stamped = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, rclpy.time.Time(),
                    timeout=Duration(seconds=self.tf_timeout_sec)
                )
                return transform_to_matrix(tf_stamped)
        except Exception:
            return None

    def _radius_outlier_removal(self, pts: np.ndarray) -> np.ndarray:
        if pts.shape[0] == 0:
            return pts
        tree = cKDTree(pts)
        neighbors = tree.query_ball_point(pts, r=self.ror_radius)
        mask = np.array([len(n) >= self.ror_min_neighbors for n in neighbors], dtype=bool)
        return pts[mask]

    def cb(self, msg: PointCloud2):
        pts = self._pc2_to_xyz(msg)
        if pts.shape[0] == 0:
            return

        pts = self._range_filter(pts)
        if pts.shape[0] == 0:
            self.pub.publish(create_pointcloud2_from_xyz(np.empty((0, 3), np.float32), msg.header.frame_id, msg.header.stamp))
            return

        pts_ds = voxel_downsample(pts, self.voxel_size, max_points=self.max_points)

        if self.prev_points is None:
            self.prev_points = pts_ds.copy()
            self.prev_frame_id = msg.header.frame_id
            self.prev_stamp = msg.header.stamp
            self.pub.publish(create_pointcloud2_from_xyz(np.empty((0, 3), np.float32), msg.header.frame_id, msg.header.stamp))
            return

        # ★重要：frame_idが同じでもTFで“時刻差”があるなら補正したいので常に試す
        prev_aligned = self.prev_points
        mat = self._try_get_tf_mat(msg.header.frame_id, self.prev_frame_id, self.prev_stamp)
        if mat is not None:
            prev_aligned = apply_transform(self.prev_points, mat)

        tree = cKDTree(prev_aligned)
        dists, _ = tree.query(pts_ds, k=1)

        z = pts_ds[:, 2]  # optical想定: 前方=z
        thr = self.thr_base + self.thr_slope * np.clip(z, 0.0, self.max_distance)

        dynamic = pts_ds[dists > thr]
        dynamic = self._radius_outlier_removal(dynamic)

        if dynamic.shape[0] < self.min_dynamic_points:
            dynamic = np.empty((0, 3), dtype=np.float32)

        self.pub.publish(create_pointcloud2_from_xyz(dynamic, msg.header.frame_id, msg.header.stamp))

        self.prev_points = pts_ds.copy()
        self.prev_frame_id = msg.header.frame_id
        self.prev_stamp = msg.header.stamp

def main(args=None):
    rclpy.init(args=args)
    node = EgoCompensator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
