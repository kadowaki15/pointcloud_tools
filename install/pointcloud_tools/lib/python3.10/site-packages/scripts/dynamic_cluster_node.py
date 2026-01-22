#!/usr/bin/env python3
"""
dynamic_static_cluster_node.py (improved)

- Robust PointCloud2 -> numpy conversion
- Faster Euclidean clustering using cKDTree region-growing (default)
- Thread-safe handling of dynamic/static callbacks
- Always publish MarkerArray (empty to clear RViz)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from scipy.spatial import cKDTree
import threading
import time

# Colors
DYNAMIC_COLOR = (1.0, 0.0, 0.0, 0.6)  # red
STATIC_COLOR = (0.0, 1.0, 0.0, 0.6)   # green

def voxel_downsample_numpy(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0.0 or points.shape[0] == 0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int64)
    keys, inv = np.unique(coords, axis=0, return_inverse=True)
    out = np.zeros((keys.shape[0], 3), dtype=np.float32)
    for i in range(keys.shape[0]):
        out[i] = points[inv == i].mean(axis=0)
    return out

def make_bbox_marker(frame_id: str, stamp, marker_id: int, center: np.ndarray, size: np.ndarray, color: tuple, ns="clusters"):
    m = Marker()
    m.header = Header()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = marker_id
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = float(center[0])
    m.pose.position.y = float(center[1])
    m.pose.position.z = float(center[2])
    m.pose.orientation.x = 0.0
    m.pose.orientation.y = 0.0
    m.pose.orientation.z = 0.0
    m.pose.orientation.w = 1.0
    m.scale.x = float(max(size[0], 0.02))
    m.scale.y = float(max(size[1], 0.02))
    m.scale.z = float(max(size[2], 0.02))
    m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
    # short lifetime to auto-clear if node dies
    m.lifetime.sec = 0
    m.lifetime.nanosec = 500_000_000
    return m

class DynamicStaticClusterNode(Node):
    def __init__(self):
        super().__init__('dynamic_static_cluster_node')

        # Parameters
        self.declare_parameter('dynamic_topic', '/pointcloud_diff_comp')
        self.declare_parameter('static_topic', '/pointcloud_static')
        self.declare_parameter('out_topic', '/dynamic_markers')
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('cluster_method', 'region_growing')  # or 'dbscan'
        self.declare_parameter('cluster_eps', 0.05)   # for DBSCAN; also used as radius in region-growing
        self.declare_parameter('cluster_min_samples', 5)
        self.declare_parameter('min_cluster_size', 40)
        self.declare_parameter('max_clusters', 30)
        self.declare_parameter('publish_empty', True)

        self.dynamic_topic = str(self.get_parameter('dynamic_topic').value)
        self.static_topic = str(self.get_parameter('static_topic').value)
        self.out_topic = str(self.get_parameter('out_topic').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.cluster_method = str(self.get_parameter('cluster_method').value)
        self.cluster_eps = float(self.get_parameter('cluster_eps').value)
        self.cluster_min_samples = int(self.get_parameter('cluster_min_samples').value)
        self.min_cluster_size = int(self.get_parameter('min_cluster_size').value)
        self.max_clusters = int(self.get_parameter('max_clusters').value)
        self.publish_empty = bool(self.get_parameter('publish_empty').value)

        # Subscriptions
        self.sub_dynamic = self.create_subscription(PointCloud2, self.dynamic_topic, self.cb_dynamic, 10)
        self.sub_static = self.create_subscription(PointCloud2, self.static_topic, self.cb_static, 10)

        # Publisher
        self.pub = self.create_publisher(MarkerArray, self.out_topic, 10)

        # Latest point clouds and header (thread-safe)
        self._lock = threading.Lock()
        self.dynamic_points = None
        self.static_points = None
        self.latest_header = None

        self.get_logger().info(f"Node started. dynamic:{self.dynamic_topic}, static:{self.static_topic}, out:{self.out_topic}")
        self.get_logger().info(f"Params: voxel={self.voxel_size} method={self.cluster_method} eps={self.cluster_eps} min_pts={self.cluster_min_samples}")

    def _pc2_to_xyz_array(self, pc2_msg: PointCloud2):
        # Robust conversion: handle structured arrays or generators
        pts_gen = pc2.read_points(pc2_msg, field_names=("x","y","z"), skip_nans=True)
        try:
            # structured numpy array case
            if hasattr(pts_gen, 'dtype') and isinstance(getattr(pts_gen, 'dtype'), np.dtype):
                dtype = pts_gen.dtype
                if dtype.names and set(('x','y','z')).issubset(dtype.names):
                    x = np.asarray(pts_gen['x'], dtype=np.float32)
                    y = np.asarray(pts_gen['y'], dtype=np.float32)
                    z = np.asarray(pts_gen['z'], dtype=np.float32)
                    return np.vstack((x, y, z)).T
            pts_list = list(pts_gen)
            if len(pts_list) == 0:
                return np.zeros((0,3), dtype=np.float32)
            arr = np.array([[p[0], p[1], p[2]] for p in pts_list], dtype=np.float32)
            return arr
        except Exception as e:
            self.get_logger().warning(f"_pc2_to_xyz_array fallback: {e}")
            pts_list = []
            for p in pc2.read_points(pc2_msg, field_names=("x","y","z"), skip_nans=True):
                try:
                    pts_list.append((float(p[0]), float(p[1]), float(p[2])))
                except Exception:
                    continue
            if len(pts_list) == 0:
                return np.zeros((0,3), dtype=np.float32)
            return np.array(pts_list, dtype=np.float32)

    def cb_dynamic(self, msg: PointCloud2):
        pts = self._pc2_to_xyz_array(msg)
        with self._lock:
            self.dynamic_points = pts
            self.latest_header = msg.header
        self.publish_markers_threadsafe()

    def cb_static(self, msg: PointCloud2):
        pts = self._pc2_to_xyz_array(msg)
        with self._lock:
            self.static_points = pts
            self.latest_header = msg.header
        self.publish_markers_threadsafe()

    def publish_markers_threadsafe(self):
        # gather snapshot under lock
        with self._lock:
            dyn = None if self.dynamic_points is None else self.dynamic_points.copy()
            sta = None if self.static_points is None else self.static_points.copy()
            header = None if self.latest_header is None else self.latest_header

        if header is None:
            # nothing to publish yet
            return

        markers = MarkerArray()

        # dynamic clusters
        if dyn is not None and dyn.shape[0] > 0:
            d_markers = self.cluster_and_make_markers(dyn, DYNAMIC_COLOR, header.frame_id, header.stamp)
            markers.markers.extend(d_markers)
            self.get_logger().debug(f"Dynamic clusters: {len(d_markers)} from {dyn.shape[0]} pts")
        # static clusters
        if sta is not None and sta.shape[0] > 0:
            s_markers = self.cluster_and_make_markers(sta, STATIC_COLOR, header.frame_id, header.stamp, ns="static")
            markers.markers.extend(s_markers)
            self.get_logger().debug(f"Static clusters: {len(s_markers)} from {sta.shape[0]} pts")

        # if none and publish_empty True, publish empty to clear RViz
        if len(markers.markers) == 0 and not self.publish_empty:
            return

        # Ensure we don't publish excessive number of markers (trim if needed)
        if len(markers.markers) > self.max_clusters:
            markers.markers = markers.markers[:self.max_clusters]

        self.pub.publish(markers)
        self.get_logger().info(f"Published {len(markers.markers)} markers")

    def cluster_and_make_markers(self, points, color, frame_id, stamp, ns="clusters"):
        # downsample
        pts_ds = voxel_downsample_numpy(points, self.voxel_size)
        if pts_ds.shape[0] == 0:
            return []

        # choose method
        if self.cluster_method == 'dbscan':
            # lazy import to avoid heavy dependency if not used
            try:
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples)
                labels = clustering.fit_predict(pts_ds)
                unique_labels = set(labels)
                markers = []
                marker_id = 0
                for label in unique_labels:
                    if label == -1:
                        continue
                    idx = np.where(labels == label)[0]
                    if len(idx) < self.min_cluster_size:
                        continue
                    cluster_pts = pts_ds[idx]
                    min_pt = cluster_pts.min(axis=0)
                    max_pt = cluster_pts.max(axis=0)
                    center = (min_pt + max_pt)/2.0
                    size = max_pt - min_pt
                    markers.append(make_bbox_marker(frame_id, stamp, marker_id, center, size, color, ns=ns))
                    marker_id += 1
                    if marker_id >= self.max_clusters:
                        break
                return markers
            except Exception as e:
                self.get_logger().warning(f"DBSCAN failed: {e}. Falling back to region_growing.")
                method = 'region_growing'
        # default: region-growing using cKDTree
        # region-growing: flood-fill neighbours within radius cluster_eps
        tree = cKDTree(pts_ds)
        N = pts_ds.shape[0]
        visited = np.zeros(N, dtype=bool)
        markers = []
        marker_id = 0
        for i in range(N):
            if visited[i]:
                continue
            # BFS/stack
            stack = [i]
            visited[i] = True
            members = []
            while stack:
                u = stack.pop()
                members.append(u)
                neigh = tree.query_ball_point(pts_ds[u], r=self.cluster_eps)
                for v in neigh:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            if len(members) < self.min_cluster_size:
                continue
            cluster_pts = pts_ds[members]
            min_pt = cluster_pts.min(axis=0)
            max_pt = cluster_pts.max(axis=0)
            center = (min_pt + max_pt)/2.0
            size = max_pt - min_pt
            markers.append(make_bbox_marker(frame_id, stamp, marker_id, center, size, color, ns=ns))
            marker_id += 1
            if marker_id >= self.max_clusters:
                break
        return markers

def main(args=None):
    rclpy.init(args=args)
    node = DynamicStaticClusterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
