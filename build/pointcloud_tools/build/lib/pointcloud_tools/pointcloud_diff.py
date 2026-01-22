#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Vector3, Pose
import numpy as np
import time

# sensor_msgs_py point_cloud2 helper
try:
    from sensor_msgs_py import point_cloud2 as pc2
except Exception:
    import sensor_msgs.point_cloud2 as pc2

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# ------------------- Parameters -------------------
TOPIC_IN = "/camera/camera/depth/color/points"
TOPIC_DIFF = "/pointcloud_diff"
TOPIC_MARKERS = "/dynamic_markers"

VOXEL_SIZE = 0.03        # m
DIFF_DIST_TH = 0.08      # m: 最近傍距離がこれ以上なら差分点
DBSCAN_EPS = 0.2         # m
DBSCAN_MIN_SAMPLES = 6

TTC_RED = 1.0            # s
TTC_YELLOW = 3.0         # s

MAX_PTS = 80000
# ---------------------------------------------------

def pc2_to_xyz_array(pc2_msg, max_pts=MAX_PTS):
    """PointCloud2 -> (N,3) numpy array"""
    pts = []
    for p in pc2.read_points(pc2_msg, field_names=("x","y","z"), skip_nans=True):
        pts.append([p[0], p[1], p[2]])
        if len(pts) >= max_pts:
            break
    if len(pts) == 0:
        return np.zeros((0,3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)

def create_pc2_from_xyz(xyz, frame_id="camera_link"):
    """(N,3) -> PointCloud2"""
    header = Header()
    header.stamp = rclpy.clock.Clock().now().to_msg()
    header.frame_id = frame_id
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    pts_list = [tuple(p.tolist()) for p in xyz] if xyz is not None else []
    return pc2.create_cloud(header, fields, pts_list)

def voxel_downsample(xyz, voxel_size):
    if xyz.shape[0] == 0:
        return xyz
    ids = np.floor(xyz / voxel_size).astype(np.int64)
    uniq, inv = np.unique(ids, axis=0, return_inverse=True)
    pts = np.zeros((uniq.shape[0], 3), dtype=np.float32)
    counts = np.zeros((uniq.shape[0],), dtype=np.int32)
    for i, idx in enumerate(inv):
        pts[idx] += xyz[i]
        counts[idx] += 1
    pts /= counts[:, None]
    return pts

class PointCloudDiffNode(Node):
    def __init__(self):
        super().__init__('pointcloud_diff_node')
        self.sub = self.create_subscription(
            PointCloud2, TOPIC_IN, self.cb_pc, 10)
        self.pub_diff = self.create_publisher(PointCloud2, TOPIC_DIFF, 1)
        self.pub_markers = self.create_publisher(MarkerArray, TOPIC_MARKERS, 1)
        self.prev_pts = None
        self.prev_time = None
        self.get_logger().info(f"Started pointcloud_diff_node subscribing {TOPIC_IN}")

    def cb_pc(self, msg: PointCloud2):
        t_now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 if msg.header.stamp else time.time()
        cur_xyz = pc2_to_xyz_array(msg)
        if cur_xyz.shape[0] == 0:
            return
        cur_xyz = voxel_downsample(cur_xyz, VOXEL_SIZE)

        if self.prev_pts is None:
            self.prev_pts = cur_xyz
            self.prev_time = t_now
            return

        dt = max(1e-3, t_now - self.prev_time)

        # 最近傍検索
        try:
            nbr = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.prev_pts)
            dists, inds = nbr.kneighbors(cur_xyz)
            dists = dists.ravel()
        except Exception as e:
            self.get_logger().warn(f"NearestNeighbors failed: {e}")
            self.prev_pts = cur_xyz
            self.prev_time = t_now
            return

        diff_mask = dists > DIFF_DIST_TH
        diff_pts = cur_xyz[diff_mask]

        # publish diff pointcloud
        self.pub_diff.publish(create_pc2_from_xyz(diff_pts, frame_id=msg.header.frame_id))

        # clustering & markers
        markers = MarkerArray()
        if diff_pts.shape[0] > 0:
            clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(diff_pts)
            labels = clustering.labels_
            unique_labels = [lab for lab in set(labels) if lab != -1]
            marker_id = 0
            for lab in unique_labels:
                cluster_pts = diff_pts[labels == lab]
                centroid = cluster_pts.mean(axis=0)

                # marker
                m = Marker()
                m.header.frame_id = msg.header.frame_id
                m.header.stamp = rclpy.clock.Clock().now().to_msg()
                m.ns = "dynamic_clusters"
                m.id = marker_id
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose = Pose()
                m.pose.position.x = float(centroid[0])
                m.pose.position.y = float(centroid[1])
                m.pose.position.z = float(centroid[2])
                m.pose.orientation.w = 1.0

                spread = float(np.linalg.norm(cluster_pts.ptp(axis=0))) if cluster_pts.shape[0] > 1 else 0.2
                scale = max(0.12, min(0.6, spread))
                m.scale = Vector3(x=scale, y=scale, z=scale)

                # color (赤=急接近、黄=接近、緑=その他)
                dist = np.linalg.norm(centroid)
                displacement = centroid - self.prev_pts[nbr.kneighbors(centroid.reshape(1,-1))[1][0,0]]
                v_rel = float(np.dot(displacement, centroid / (dist + 1e-8)) / dt) if dist > 0 else 0.0
                ttc = dist / v_rel if v_rel > 0.01 else float('inf')
                if ttc < TTC_RED:
                    m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)
                elif ttc < TTC_YELLOW:
                    m.color = ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.85)
                elif ttc != float('inf'):
                    m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.6)
                else:
                    m.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.5)

                markers.markers.append(m)
                marker_id += 1

        self.pub_markers.publish(markers)

        # update previous
        self.prev_pts = cur_xyz
        self.prev_time = t_now

def main():
    rclpy.init()
    node = PointCloudDiffNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
