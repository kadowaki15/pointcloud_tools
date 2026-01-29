#!/usr/bin/env python3
"""
dynamic_static_cluster_node.py (PIPELINE-SAFE FINAL)

目的:
- /pointcloud_diff_comp (dynamic) と /pointcloud_static (static) をクラスタリングして bbox MarkerArray を publish
- tracker が拾えるように dynamic は ns="clusters" を固定
- dynamic/static で header(stamp) を分離（混線防止）
- ROI + (任意)距離適応min_clusterで clip_distance 後でもクラスタが成立しやすい

注意:
- ここでは “approach-only” フィルタは入れない（入れると crossing は出なくなる）
- approach/crossing の分類は intent_estimator に任せる
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


DYNAMIC_COLOR = (1.0, 0.0, 0.0, 0.6)  # red
STATIC_COLOR  = (0.0, 1.0, 0.0, 0.6)  # green


def voxel_downsample_numpy(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0.0 or points.shape[0] == 0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int64)
    keys, inv = np.unique(coords, axis=0, return_inverse=True)
    out = np.zeros((keys.shape[0], 3), dtype=np.float32)
    for i in range(keys.shape[0]):
        out[i] = points[inv == i].mean(axis=0)
    return out


def make_bbox_marker(frame_id: str, stamp, marker_id: int,
                     center: np.ndarray, size: np.ndarray,
                     color: tuple, ns: str):
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
    m.pose.orientation.w = 1.0

    m.scale.x = float(max(size[0], 0.02))
    m.scale.y = float(max(size[1], 0.02))
    m.scale.z = float(max(size[2], 0.02))

    m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])

    m.lifetime.sec = 0
    m.lifetime.nanosec = 500_000_000
    return m


class DynamicStaticClusterNode(Node):
    def __init__(self):
        super().__init__('dynamic_static_cluster_node')

        # Topics
        self.declare_parameter('dynamic_topic', '/pointcloud_diff_comp')
        self.declare_parameter('static_topic',  '/pointcloud_static')
        self.declare_parameter('out_topic',     '/dynamic_markers')

        # Downsample / clustering
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('cluster_method', 'region_growing')  # 'region_growing' or 'dbscan'
        self.declare_parameter('cluster_eps', 0.08)
        self.declare_parameter('cluster_min_samples', 5)  # DBSCAN only
        self.declare_parameter('min_cluster_size', 30)
        self.declare_parameter('max_clusters', 30)
        self.declare_parameter('publish_empty', True)

        # ROI
        self.declare_parameter('enable_roi', True)
        self.declare_parameter('roi_x_min', -0.8)
        self.declare_parameter('roi_x_max',  0.8)
        self.declare_parameter('roi_y_min', -0.5)
        self.declare_parameter('roi_y_max',  0.5)
        self.declare_parameter('roi_z_min',  0.2)
        self.declare_parameter('roi_z_max',  2.0)

        # axis mapping for ROI / forward distance (for adaptive min_cluster)
        self.declare_parameter('roi_axis', 'xyz')  # optical想定: forward=z

        # adaptive min cluster
        self.declare_parameter('enable_adaptive_min_cluster', True)
        self.declare_parameter('adapt_z0', 0.8)
        self.declare_parameter('adapt_z1', 1.5)
        self.declare_parameter('adapt_min0', 40)
        self.declare_parameter('adapt_min1', 28)
        self.declare_parameter('adapt_min2', 18)

        self._load_params()

        # Subs/Pub
        self.sub_dynamic = self.create_subscription(PointCloud2, self.dynamic_topic, self.cb_dynamic, 10)
        self.sub_static  = self.create_subscription(PointCloud2, self.static_topic,  self.cb_static,  10)
        self.pub = self.create_publisher(MarkerArray, self.out_topic, 10)

        # Thread-safe cache (IMPORTANT: split headers)
        self._lock = threading.Lock()
        self.dynamic_points = None
        self.static_points  = None
        self.dynamic_header = None
        self.static_header  = None

        self.get_logger().info(
            f"Started. dynamic:{self.dynamic_topic}, static:{self.static_topic}, out:{self.out_topic}"
        )
        self.get_logger().info(
            f"Params: voxel={self.voxel_size} eps={self.cluster_eps} "
            f"min_cluster={self.min_cluster_size} ROI={self.enable_roi} axis={self.roi_axis} "
            f"adaptive={self.enable_adaptive_min_cluster}"
        )
        self.get_logger().info(
            "NOTE: dynamic markers ns is fixed to 'clusters' for tracker compatibility."
        )

    def _load_params(self):
        self.dynamic_topic = str(self.get_parameter('dynamic_topic').value)
        self.static_topic  = str(self.get_parameter('static_topic').value)
        self.out_topic     = str(self.get_parameter('out_topic').value)

        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.cluster_method = str(self.get_parameter('cluster_method').value)
        self.cluster_eps = float(self.get_parameter('cluster_eps').value)
        self.cluster_min_samples = int(self.get_parameter('cluster_min_samples').value)
        self.min_cluster_size = int(self.get_parameter('min_cluster_size').value)
        self.max_clusters = int(self.get_parameter('max_clusters').value)
        self.publish_empty = bool(self.get_parameter('publish_empty').value)

        self.enable_roi = bool(self.get_parameter('enable_roi').value)
        self.roi_x_min = float(self.get_parameter('roi_x_min').value)
        self.roi_x_max = float(self.get_parameter('roi_x_max').value)
        self.roi_y_min = float(self.get_parameter('roi_y_min').value)
        self.roi_y_max = float(self.get_parameter('roi_y_max').value)
        self.roi_z_min = float(self.get_parameter('roi_z_min').value)
        self.roi_z_max = float(self.get_parameter('roi_z_max').value)

        self.roi_axis = str(self.get_parameter('roi_axis').value)
        if set(self.roi_axis) != set("xyz") or len(self.roi_axis) != 3:
            self.get_logger().warning("roi_axis invalid. Falling back to 'xyz'.")
            self.roi_axis = "xyz"

        self.enable_adaptive_min_cluster = bool(self.get_parameter('enable_adaptive_min_cluster').value)
        self.adapt_z0 = float(self.get_parameter('adapt_z0').value)
        self.adapt_z1 = float(self.get_parameter('adapt_z1').value)
        self.adapt_min0 = int(self.get_parameter('adapt_min0').value)
        self.adapt_min1 = int(self.get_parameter('adapt_min1').value)
        self.adapt_min2 = int(self.get_parameter('adapt_min2').value)

    # ---------- PointCloud conversion ----------
    def _pc2_to_xyz_array(self, pc2_msg: PointCloud2) -> np.ndarray:
        pts_gen = pc2.read_points(pc2_msg, field_names=("x", "y", "z"), skip_nans=True)
        try:
            pts_list = list(pts_gen)
            if len(pts_list) == 0:
                return np.zeros((0, 3), dtype=np.float32)
            return np.array([[p[0], p[1], p[2]] for p in pts_list], dtype=np.float32)
        except Exception as e:
            self.get_logger().warning(f"_pc2_to_xyz_array fallback: {e}")
            pts_list = []
            for p in pc2.read_points(pc2_msg, field_names=("x", "y", "z"), skip_nans=True):
                try:
                    pts_list.append((float(p[0]), float(p[1]), float(p[2])))
                except Exception:
                    continue
            if len(pts_list) == 0:
                return np.zeros((0, 3), dtype=np.float32)
            return np.array(pts_list, dtype=np.float32)

    # ---------- ROI ----------
    def _apply_roi(self, points: np.ndarray) -> np.ndarray:
        if (not self.enable_roi) or points.shape[0] == 0:
            return points

        idx_map = {'x': 0, 'y': 1, 'z': 2}
        ordered = points[:, [idx_map[self.roi_axis[0]], idx_map[self.roi_axis[1]], idx_map[self.roi_axis[2]]]]

        m = (
            (ordered[:, 0] > self.roi_x_min) & (ordered[:, 0] < self.roi_x_max) &
            (ordered[:, 1] > self.roi_y_min) & (ordered[:, 1] < self.roi_y_max) &
            (ordered[:, 2] > self.roi_z_min) & (ordered[:, 2] < self.roi_z_max)
        )
        return points[m]

    def _forward_distance_of_center(self, center_xyz: np.ndarray) -> float:
        idx_map = {'x': 0, 'y': 1, 'z': 2}
        # logical z (forward) is roi_axis[2]
        return float(center_xyz[idx_map[self.roi_axis[2]]])

    def _required_cluster_size(self, forward_dist: float) -> int:
        if not self.enable_adaptive_min_cluster:
            return self.min_cluster_size
        if forward_dist < self.adapt_z0:
            return self.adapt_min0
        if forward_dist < self.adapt_z1:
            return self.adapt_min1
        return self.adapt_min2

    # ---------- Callbacks ----------
    def cb_dynamic(self, msg: PointCloud2):
        pts = self._pc2_to_xyz_array(msg)
        with self._lock:
            self.dynamic_points = pts
            self.dynamic_header = msg.header
        self.publish_markers_threadsafe()

    def cb_static(self, msg: PointCloud2):
        pts = self._pc2_to_xyz_array(msg)
        with self._lock:
            self.static_points = pts
            self.static_header = msg.header
        self.publish_markers_threadsafe()

    # ---------- Publish ----------
    def publish_markers_threadsafe(self):
        with self._lock:
            dyn = None if self.dynamic_points is None else self.dynamic_points.copy()
            sta = None if self.static_points  is None else self.static_points.copy()
            dyn_header = self.dynamic_header
            sta_header = self.static_header

        markers = MarkerArray()

        # ★dynamic: ns must be "clusters"
        if dyn is not None and dyn.shape[0] > 0 and dyn_header is not None:
            d_markers = self.cluster_and_make_markers(
                dyn, DYNAMIC_COLOR, dyn_header.frame_id, dyn_header.stamp, ns="clusters"
            )
            markers.markers.extend(d_markers)

        # static: keep "static" (or change if your downstream expects something else)
        if sta is not None and sta.shape[0] > 0 and sta_header is not None:
            s_markers = self.cluster_and_make_markers(
                sta, STATIC_COLOR, sta_header.frame_id, sta_header.stamp, ns="static"
            )
            markers.markers.extend(s_markers)

        if len(markers.markers) == 0 and not self.publish_empty:
            return

        if len(markers.markers) > self.max_clusters:
            markers.markers = markers.markers[:self.max_clusters]

        self.pub.publish(markers)

    # ---------- Clustering ----------
    def cluster_and_make_markers(self, points: np.ndarray, color: tuple, frame_id: str, stamp, ns: str):
        points = self._apply_roi(points)
        if points.shape[0] == 0:
            return []

        pts_ds = voxel_downsample_numpy(points, self.voxel_size)
        if pts_ds.shape[0] == 0:
            return []

        # DBSCAN optional
        if self.cluster_method == 'dbscan':
            try:
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples)
                labels = clustering.fit_predict(pts_ds)

                markers = []
                marker_id = 0
                for label in set(labels):
                    if label == -1:
                        continue
                    idx = np.where(labels == label)[0]
                    cluster_pts = pts_ds[idx]
                    min_pt = cluster_pts.min(axis=0)
                    max_pt = cluster_pts.max(axis=0)
                    center = (min_pt + max_pt) / 2.0

                    req = self._required_cluster_size(self._forward_distance_of_center(center))
                    if len(idx) < req:
                        continue

                    size = max_pt - min_pt
                    markers.append(make_bbox_marker(frame_id, stamp, marker_id, center, size, color, ns=ns))
                    marker_id += 1
                    if marker_id >= self.max_clusters:
                        break
                return markers
            except Exception as e:
                self.get_logger().warning(f"DBSCAN failed: {e}. Falling back to region_growing.")

        # region-growing
        tree = cKDTree(pts_ds)
        N = pts_ds.shape[0]
        visited = np.zeros(N, dtype=bool)

        markers = []
        marker_id = 0

        for i in range(N):
            if visited[i]:
                continue
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

            cluster_pts = pts_ds[members]
            min_pt = cluster_pts.min(axis=0)
            max_pt = cluster_pts.max(axis=0)
            center = (min_pt + max_pt) / 2.0

            req = self._required_cluster_size(self._forward_distance_of_center(center))
            if len(members) < req:
                continue

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
