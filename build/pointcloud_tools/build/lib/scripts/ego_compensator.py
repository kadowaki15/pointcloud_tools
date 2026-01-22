#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
from scipy.spatial import cKDTree as KDTree
from std_msgs.msg import Header
import time
from sklearn.neighbors import NearestNeighbors

# 4x4 行列に変換
def transform_to_matrix(tf_stamped: TransformStamped):
    t = tf_stamped.transform.translation
    q = tf_stamped.transform.rotation
    tx, ty, tz = t.x, t.y, t.z
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n > 0:
        qx/=n; qy/=n; qz/=n; qw/=n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    rot = np.array([
        [1-2*(yy+zz), 2*(xy-wz), 2*(xz+wy)],
        [2*(xy+wz), 1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy)]
    ], dtype=np.float32)
    mat = np.eye(4, dtype=np.float32)
    mat[:3,:3] = rot
    mat[:3,3] = np.array([tx, ty, tz], dtype=np.float32)
    return mat

def apply_transform(points: np.ndarray, mat: np.ndarray):
    if points.size==0: return points
    N = points.shape[0]
    hom = np.ones((N,4), dtype=np.float32)
    hom[:,:3] = points
    return (hom.dot(mat.T))[:,:3]

def create_pointcloud2_from_xyz(points_xyz: np.ndarray, frame_id: str, stamp=None):
    header = Header()
    header.frame_id = frame_id
    if stamp is None:
        header.stamp = rclpy.time.Time(seconds=int(time.time())).to_msg()
    else:
        header.stamp = stamp
    return pc2.create_cloud_xyz32(header, points_xyz.tolist())

class EgoCompensator(Node):
    def __init__(self):
        super().__init__('ego_compensator')

        self.distance_threshold = 0.05  # 動的判定距離
        self.max_points = 50000         # サンプリング上限
        self.max_distance = 3.0         # 遠距離フィルタ (m)
        self.outlier_radius = 0.1       # 孤立点除去の半径
        self.min_neighbors = 3          # 孤立点除去の最小近傍数

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.prev_points = None
        self.prev_frame_id = None
        self.prev_frame_time = None

        self.sub = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.cb,
            10
        )
        self.pub = self.create_publisher(PointCloud2, '/pointcloud_diff_comp', 10)

        self.get_logger().info("EgoCompensator with noise filtering started")

    def cb(self, msg):
        try:
            # 点群を Nx3 に変換
            pts_list = list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True))
            if len(pts_list)==0:
                return
            pts = np.array([[p[0],p[1],p[2]] for p in pts_list], dtype=np.float32)

            # サンプリング
            if pts.shape[0] > self.max_points:
                idx = np.random.choice(pts.shape[0], self.max_points, replace=False)
                pts_sub = pts[idx]
            else:
                pts_sub = pts

            # 前フレームがない場合は保存して終了
            if self.prev_points is None:
                self.prev_points = pts_sub.copy()
                self.prev_frame_id = msg.header.frame_id
                self.prev_frame_time = msg.header.stamp
                empty_pc = create_pointcloud2_from_xyz(np.empty((0,3),dtype=np.float32),
                                                       msg.header.frame_id, msg.header.stamp)
                self.pub.publish(empty_pc)
                return

            # TF で前フレームを現在フレーム座標に変換
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    msg.header.frame_id,
                    self.prev_frame_id,
                    rclpy.time.Time(seconds=self.prev_frame_time.sec,
                                   nanoseconds=self.prev_frame_time.nanosec),
                    timeout=rclpy.duration.Duration(seconds=0.05)
                )
                transform_mat = transform_to_matrix(transform_stamped)
            except Exception:
                transform_mat = np.eye(4, dtype=np.float32)

            prev_transformed = apply_transform(self.prev_points, transform_mat)

            # KDTree で最近傍探索
            tree = KDTree(prev_transformed)
            dists, idxs = tree.query(pts_sub, k=1, n_jobs=-1)
            dynamic_pts = pts_sub[dists > self.distance_threshold]

            # 遠距離フィルタ
            dist_from_origin = np.linalg.norm(dynamic_pts, axis=1)
            dynamic_pts = dynamic_pts[dist_from_origin < self.max_distance]

            # 孤立点除去 (Radius Outlier Removal)
            if dynamic_pts.shape[0] > 0:
                nbrs = NearestNeighbors(radius=self.outlier_radius).fit(dynamic_pts)
                neighbors = nbrs.radius_neighbors(dynamic_pts, return_distance=False)
                mask = np.array([len(n) >= self.min_neighbors for n in neighbors])
                dynamic_pts = dynamic_pts[mask]

            # 出力
            pc_out = create_pointcloud2_from_xyz(dynamic_pts, msg.header.frame_id, msg.header.stamp)
            self.pub.publish(pc_out)

            # 前フレーム更新
            self.prev_points = pts_sub.copy()
            self.prev_frame_id = msg.header.frame_id
            self.prev_frame_time = msg.header.stamp

        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")

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
