#!/usr/bin/env python3
"""
dynamic_cluster_marker.py

Usage:
  python3 scripts/dynamic_cluster_marker.py

What it does:
 - subscribe to /pointcloud_diff (sensor_msgs/PointCloud2)
 - downsample points by voxel grid (voxel_size param)
 - cluster points using DBSCAN (dbscan_eps, dbscan_min_samples)
 - filter small clusters (min_cluster_size)
 - publish MarkerArray (/dynamic_markers) with CUBE markers (bounding boxes) for each cluster

Notes:
 - Requires scikit-learn. Install if missing:
     python3 -m pip install scikit-learn
"""
import sys
import math
import traceback
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

# Try to import DBSCAN from sklearn; if missing, instruct user to install
try:
    from sklearn.cluster import DBSCAN
except Exception:
    DBSCAN = None

# Utility: deterministic color palette
DEFAULT_COLORS = [
    (0.0, 1.0, 0.0, 0.6),
    (0.0, 0.0, 1.0, 0.6),
    (1.0, 0.0, 0.0, 0.6),
    (1.0, 1.0, 0.0, 0.6),
    (1.0, 0.0, 1.0, 0.6),
    (0.0, 1.0, 1.0, 0.6),
    (0.8, 0.4, 0.0, 0.6),
    (0.4, 0.0, 0.8, 0.6),
]

def voxel_downsample_numpy(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Simple voxel downsample. points: (N,3), returns downsampled points (M,3)
    """
    if voxel_size <= 0.0:
        return points
    # Compute voxel indices
    coords = np.floor(points / voxel_size).astype(np.int64)
    # Create a viewable tuple key
    # Unique voxels and centroids
    # Use structured view to unique rows
    if coords.size == 0:
        return points
    # Convert to 1D keys by hashing tuples (fast approach)
    # But to ensure deterministic grouping, use lexsort unique
    keys, inv = np.unique(coords, axis=0, return_inverse=True)
    out = np.zeros((keys.shape[0], 3), dtype=np.float32)
    for i in range(keys.shape[0]):
        out[i, :] = points[inv == i].mean(axis=0)
    return out

def make_bbox_marker(frame_id: str, stamp, marker_id: int, center: np.ndarray, size: np.ndarray, color: tuple, ns="dynamic_clusters"):
    m = Marker()
    m.header = Header()
    m.header.frame_id = frame_id
    # stamp may be Time or builtin; set to now if None
    try:
        m.header.stamp = stamp
    except Exception:
        # create a ROS time now
        from rclpy.time import Time
        m.header.stamp = Time(seconds=0).to_msg()
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
    # ensure min size
    min_size = 0.02
    sx = float(max(size[0], min_size))
    sy = float(max(size[1], min_size))
    sz = float(max(size[2], min_size))
    m.scale.x = sx
    m.scale.y = sy
    m.scale.z = sz
    m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
    # short lifetime to avoid stale markers lingering; tracker can republish
    m.lifetime.sec = 0
    m.lifetime.nanosec = 500000000  # 0.5 sec
    return m

class DynamicClusterMarker(Node):
    def __init__(self):
        super().__init__('dynamic_cluster_marker')

        # Parameters (tweak or change via --ros-args -p ...)
        self.declare_parameter('in_topic', '/pointcloud_diff')
        self.declare_parameter('out_topic', '/dynamic_markers')
        self.declare_parameter('voxel_size', 0.02)         # meters
        self.declare_parameter('dbscan_eps', 0.05)         # meters
        self.declare_parameter('dbscan_min_samples', 10)
        self.declare_parameter('min_cluster_size', 30)
        self.declare_parameter('max_clusters', 20)

        self.in_topic = str(self.get_parameter('in_topic').value)
        self.out_topic = str(self.get_parameter('out_topic').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.dbscan_eps = float(self.get_parameter('dbscan_eps').value)
        self.dbscan_min_samples = int(self.get_parameter('dbscan_min_samples').value)
        self.min_cluster_size = int(self.get_parameter('min_cluster_size').value)
        self.max_clusters = int(self.get_parameter('max_clusters').value)

        # subscription & publisher
        self.sub = self.create_subscription(PointCloud2, self.in_topic, self.cb, 10)
        self.pub = self.create_publisher(MarkerArray, self.out_topic, 10)

        # log
        self.get_logger().info(f"DynamicClusterMarker: sub={self.in_topic} pub={self.out_topic} voxel={self.voxel_size} eps={self.dbscan_eps} min_samples={self.dbscan_min_samples} min_cluster_size={self.min_cluster_size}")

        if DBSCAN is None:
            self.get_logger().warn("scikit-learn DBSCAN not available. Install with: python3 -m pip install scikit-learn")
            # do not exit; we will attempt and error clearly in cb

    def cb(self, msg: PointCloud2):
        # read points
        try:
            pts_gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            pts_list = list(pts_gen)
            if len(pts_list) == 0:
                # publish empty MarkerArray to clear? publish empty with no markers and rely on lifetime
                return
            pts = np.array(pts_list, dtype=np.float32)  # (N,3)
        except Exception as e:
            self.get_logger().error(f"Failed to read pointcloud: {e}\n{traceback.format_exc()}")
            return

        # Downsample
        if self.voxel_size > 0.0:
            pts_ds = voxel_downsample_numpy(pts, self.voxel_size)
        else:
            pts_ds = pts

        if pts_ds.shape[0] == 0:
            return

        # Clustering
        if DBSCAN is None:
            self.get_logger().error("DBSCAN unavailable. Please install scikit-learn: python3 -m pip install scikit-learn")
            return

        try:
            clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, metric='euclidean')
            labels = clustering.fit_predict(pts_ds)
        except Exception as e:
            self.get_logger().error(f"DBSCAN failed: {e}\n{traceback.format_exc()}")
            return

        unique_labels = [l for l in set(labels) if l != -1]
        # sort labels by cluster size descending (largest first)
        clusters = []
        for l in unique_labels:
            idx = np.where(labels == l)[0]
            clusters.append((l, idx))
        clusters.sort(key=lambda x: -len(x[1]))

        markers = MarkerArray()
        marker_id = 0
        frame_id = msg.header.frame_id if hasattr(msg, 'header') else 'camera_depth_optical_frame'
        stamp = msg.header.stamp if hasattr(msg, 'header') else None

        for (label, indices) in clusters:
            if marker_id >= self.max_clusters:
                break
            if len(indices) < self.min_cluster_size:
                continue
            cluster_pts = pts_ds[indices]
            # bounding box
            min_pt = cluster_pts.min(axis=0)
            max_pt = cluster_pts.max(axis=0)
            center = (min_pt + max_pt) / 2.0
            size = (max_pt - min_pt)
            # choose color deterministically
            color = DEFAULT_COLORS[marker_id % len(DEFAULT_COLORS)]
            m = make_bbox_marker(frame_id, stamp, marker_id, center, size, color)
            markers.markers.append(m)
            marker_id += 1

        # publish MarkerArray
        self.pub.publish(markers)

def main(args=None):
    rclpy.init(args=args)
    node = DynamicClusterMarker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
