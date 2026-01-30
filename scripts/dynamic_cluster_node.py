#!/usr/bin/env python3
"""
dynamic_static_cluster_node.py (PIPELINE-SAFE FINAL - STATIC SUPPRESS + NMS)

追加改善（あなたの症状向け）:
- Static suppression: /pointcloud_static に近い dynamic 点を除去（背景由来の差分を殺す）
- NMS: 近い/重なる dynamic bbox を「最大スコアだけ残す」（同座標付近の多重マーカー抑制）
- ROR + size/quality filter + DELETEALL は維持
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


DYNAMIC_COLOR = (1.0, 0.0, 0.0, 0.6)
STATIC_COLOR  = (0.0, 1.0, 0.0, 0.6)


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
        self.declare_parameter('voxel_size', 0.07)
        self.declare_parameter('cluster_method', 'region_growing')
        self.declare_parameter('cluster_eps', 0.09)
        self.declare_parameter('cluster_min_samples', 5)
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

        # axis mapping (optical想定: forward=z)
        self.declare_parameter('roi_axis', 'xyz')

        # adaptive min cluster
        self.declare_parameter('enable_adaptive_min_cluster', True)
        self.declare_parameter('adapt_z0', 0.8)
        self.declare_parameter('adapt_z1', 1.5)
        self.declare_parameter('adapt_min0', 40)
        self.declare_parameter('adapt_min1', 28)
        self.declare_parameter('adapt_min2', 18)

        # adaptive eps（デフォOFF）
        self.declare_parameter('enable_adaptive_eps', False)
        self.declare_parameter('eps_near', 0.09)
        self.declare_parameter('eps_mid',  0.11)
        self.declare_parameter('eps_far',  0.13)

        # ROR
        self.declare_parameter('enable_ror', True)
        self.declare_parameter('ror_radius', 0.12)
        self.declare_parameter('ror_min_neighbors', 3)

        # merge（y整合あり）
        self.declare_parameter('enable_merge', True)
        self.declare_parameter('merge_center_dist', 0.24)
        self.declare_parameter('merge_iou_thresh', 0.12)
        self.declare_parameter('merge_y_dist', 0.25)
        self.declare_parameter('pre_min_cluster_size', 15)

        # size / quality filter（デフォON）
        self.declare_parameter('enable_size_filter', True)
        self.declare_parameter('min_size_x', 0.10)
        self.declare_parameter('min_size_y', 0.10)
        self.declare_parameter('min_size_z', 0.15)

        self.declare_parameter('enable_quality_filter', True)
        self.declare_parameter('min_bbox_volume', 0.0012)
        self.declare_parameter('min_bbox_diag',   0.18)

        # ★NEW: static suppression（背景除去の本命）
        self.declare_parameter('enable_static_suppression', True)
        self.declare_parameter('static_suppress_radius', 0.06)  # 静的点の近傍を除去（5〜8cmで調整）

        # ★NEW: NMS（同座標付近を1つにする）
        self.declare_parameter('enable_nms', True)
        self.declare_parameter('nms_center_dist', 0.35)  # centerが近い箱は1つに（人が横切る用途なら0.3〜0.5）
        self.declare_parameter('nms_iou_thresh', 0.08)   # x-z IoUがこれ以上なら同一扱い
        self.declare_parameter('nms_use_xz', True)       # x-z平面で判断（optical前提）

        self._load_params()

        self.sub_dynamic = self.create_subscription(PointCloud2, self.dynamic_topic, self.cb_dynamic, 10)
        self.sub_static  = self.create_subscription(PointCloud2, self.static_topic,  self.cb_static,  10)
        self.pub = self.create_publisher(MarkerArray, self.out_topic, 10)

        self._lock = threading.Lock()
        self.dynamic_points = None
        self.static_points  = None
        self.dynamic_header = None
        self.static_header  = None

        # static KDTree cache
        self._static_ds = None
        self._static_tree = None
        self._static_frame = None

        self.get_logger().info("Started dynamic_static_cluster_node (static suppress + nms).")

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
            self.roi_axis = "xyz"

        self.enable_adaptive_min_cluster = bool(self.get_parameter('enable_adaptive_min_cluster').value)
        self.adapt_z0 = float(self.get_parameter('adapt_z0').value)
        self.adapt_z1 = float(self.get_parameter('adapt_z1').value)
        self.adapt_min0 = int(self.get_parameter('adapt_min0').value)
        self.adapt_min1 = int(self.get_parameter('adapt_min1').value)
        self.adapt_min2 = int(self.get_parameter('adapt_min2').value)

        self.enable_adaptive_eps = bool(self.get_parameter('enable_adaptive_eps').value)
        self.eps_near = float(self.get_parameter('eps_near').value)
        self.eps_mid  = float(self.get_parameter('eps_mid').value)
        self.eps_far  = float(self.get_parameter('eps_far').value)

        self.enable_ror = bool(self.get_parameter('enable_ror').value)
        self.ror_radius = float(self.get_parameter('ror_radius').value)
        self.ror_min_neighbors = int(self.get_parameter('ror_min_neighbors').value)

        self.enable_merge = bool(self.get_parameter('enable_merge').value)
        self.merge_center_dist = float(self.get_parameter('merge_center_dist').value)
        self.merge_iou_thresh  = float(self.get_parameter('merge_iou_thresh').value)
        self.merge_y_dist      = float(self.get_parameter('merge_y_dist').value)
        self.pre_min_cluster_size = int(self.get_parameter('pre_min_cluster_size').value)

        self.enable_size_filter = bool(self.get_parameter('enable_size_filter').value)
        self.min_size_x = float(self.get_parameter('min_size_x').value)
        self.min_size_y = float(self.get_parameter('min_size_y').value)
        self.min_size_z = float(self.get_parameter('min_size_z').value)

        self.enable_quality_filter = bool(self.get_parameter('enable_quality_filter').value)
        self.min_bbox_volume = float(self.get_parameter('min_bbox_volume').value)
        self.min_bbox_diag   = float(self.get_parameter('min_bbox_diag').value)

        self.enable_static_suppression = bool(self.get_parameter('enable_static_suppression').value)
        self.static_suppress_radius = float(self.get_parameter('static_suppress_radius').value)

        self.enable_nms = bool(self.get_parameter('enable_nms').value)
        self.nms_center_dist = float(self.get_parameter('nms_center_dist').value)
        self.nms_iou_thresh = float(self.get_parameter('nms_iou_thresh').value)
        self.nms_use_xz = bool(self.get_parameter('nms_use_xz').value)

    # ---------- axis ----------
    @staticmethod
    def _axis_index(axis_letter: str) -> int:
        return {'x': 0, 'y': 1, 'z': 2}[axis_letter]

    def _logical_x_index(self) -> int:
        return self._axis_index(self.roi_axis[0])

    def _logical_y_index(self) -> int:
        return self._axis_index(self.roi_axis[1])

    def _logical_z_index(self) -> int:
        return self._axis_index(self.roi_axis[2])

    # ---------- PointCloud conversion ----------
    def _pc2_to_xyz_array(self, pc2_msg: PointCloud2) -> np.ndarray:
        pts_gen = pc2.read_points(pc2_msg, field_names=("x", "y", "z"), skip_nans=True)
        pts_list = list(pts_gen)
        if len(pts_list) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array([[p[0], p[1], p[2]] for p in pts_list], dtype=np.float32)

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
        return float(center_xyz[self._logical_z_index()])

    def _required_cluster_size(self, forward_dist: float) -> int:
        if not self.enable_adaptive_min_cluster:
            return self.min_cluster_size
        if forward_dist < self.adapt_z0:
            return self.adapt_min0
        if forward_dist < self.adapt_z1:
            return self.adapt_min1
        return self.adapt_min2

    def _eps_for_forward(self, forward_dist: float) -> float:
        if not self.enable_adaptive_eps:
            return self.cluster_eps
        if forward_dist < self.adapt_z0:
            return self.eps_near
        if forward_dist < self.adapt_z1:
            return self.eps_mid
        return self.eps_far

    # ---------- ROR ----------
    def _radius_outlier_removal(self, pts: np.ndarray) -> np.ndarray:
        if (not self.enable_ror) or pts.shape[0] == 0:
            return pts
        tree = cKDTree(pts)
        r = float(self.ror_radius)
        thr = int(self.ror_min_neighbors) + 1  # include self
        keep = np.zeros((pts.shape[0],), dtype=bool)
        for i in range(pts.shape[0]):
            if len(tree.query_ball_point(pts[i], r=r)) >= thr:
                keep[i] = True
        return pts[keep]

    # ---------- static suppression ----------
    def _suppress_near_static(self, dyn_pts: np.ndarray, frame_id: str) -> np.ndarray:
        if (not self.enable_static_suppression) or dyn_pts.shape[0] == 0:
            return dyn_pts

        with self._lock:
            tree = self._static_tree
            s_frame = self._static_frame

        # frameが違うなら抑制しない（危険なので）
        if tree is None or (s_frame is not None and s_frame != frame_id):
            return dyn_pts

        # nearest neighbor distance
        d, _ = tree.query(dyn_pts, k=1, distance_upper_bound=float(self.static_suppress_radius))
        # d==inf は近傍なし。inf以外（=静的に近い）は除去
        keep = np.isinf(d)
        return dyn_pts[keep]

    # ---------- bbox helpers ----------
    def _bbox_iou_xz(self, a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
        ix = self._logical_x_index()
        iz = self._logical_z_index()
        a0 = np.array([a_min[ix], a_min[iz]], dtype=np.float32)
        a1 = np.array([a_max[ix], a_max[iz]], dtype=np.float32)
        b0 = np.array([b_min[ix], b_min[iz]], dtype=np.float32)
        b1 = np.array([b_max[ix], b_max[iz]], dtype=np.float32)
        inter_min = np.maximum(a0, b0)
        inter_max = np.minimum(a1, b1)
        inter = np.maximum(0.0, inter_max - inter_min)
        inter_area = float(inter[0] * inter[1])
        a_wh = np.maximum(0.0, a1 - a0)
        b_wh = np.maximum(0.0, b1 - b0)
        a_area = float(a_wh[0] * a_wh[1])
        b_area = float(b_wh[0] * b_wh[1])
        denom = a_area + b_area - inter_area
        return inter_area / denom if denom > 1e-9 else 0.0

    def _merge_clusters_bbox(self, clusters):
        n = len(clusters)
        if n <= 1 or (not self.enable_merge):
            return clusters

        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def unite(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        iy = self._logical_y_index()

        for i in range(n):
            ci = clusters[i]["center"]
            for j in range(i+1, n):
                cj = clusters[j]["center"]
                if abs(float(ci[iy] - cj[iy])) > self.merge_y_dist:
                    continue
                if float(np.linalg.norm(ci - cj)) < self.merge_center_dist:
                    unite(i, j); continue
                if self._bbox_iou_xz(clusters[i]["min"], clusters[i]["max"],
                                     clusters[j]["min"], clusters[j]["max"]) >= self.merge_iou_thresh:
                    unite(i, j)

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        merged = []
        for ids in groups.values():
            mins = np.min([clusters[k]["min"] for k in ids], axis=0)
            maxs = np.max([clusters[k]["max"] for k in ids], axis=0)
            nn = int(np.sum([clusters[k]["n"] for k in ids]))
            c = (mins + maxs)/2.0
            merged.append({"min": mins, "max": maxs, "center": c, "n": nn})
        return merged

    # ---------- NMS ----------
    def _nms_clusters(self, clusters):
        if (not self.enable_nms) or len(clusters) <= 1:
            return clusters

        # score: 点数優先 + 体積少し
        def score(c):
            size = c["max"] - c["min"]
            vol = float(size[0]*size[1]*size[2])
            return float(c["n"]) + 20.0*vol

        clusters = sorted(clusters, key=score, reverse=True)
        kept = []
        for c in clusters:
            ok = True
            for k in kept:
                if float(np.linalg.norm(c["center"] - k["center"])) < self.nms_center_dist:
                    ok = False; break
                if self._bbox_iou_xz(c["min"], c["max"], k["min"], k["max"]) >= self.nms_iou_thresh:
                    ok = False; break
            if ok:
                kept.append(c)
        return kept

    def _size_filter_ok(self, size_xyz: np.ndarray) -> bool:
        if not self.enable_size_filter:
            return True
        return (size_xyz[0] >= self.min_size_x) and (size_xyz[1] >= self.min_size_y) and (size_xyz[2] >= self.min_size_z)

    def _quality_filter_ok(self, size_xyz: np.ndarray) -> bool:
        if not self.enable_quality_filter:
            return True
        vol = float(size_xyz[0]*size_xyz[1]*size_xyz[2])
        diag = float(np.linalg.norm(size_xyz))
        return (vol >= self.min_bbox_volume) and (diag >= self.min_bbox_diag)

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

        # static KDTree cache update（軽くするためdownsample）
        pts_roi = self._apply_roi(pts)
        s_ds = voxel_downsample_numpy(pts_roi, self.voxel_size)
        if s_ds.shape[0] > 0:
            tree = cKDTree(s_ds)
        else:
            tree = None
        with self._lock:
            self._static_ds = s_ds
            self._static_tree = tree
            self._static_frame = msg.header.frame_id

        self.publish_markers_threadsafe()

    # ---------- Publish ----------
    def publish_markers_threadsafe(self):
        with self._lock:
            dyn = None if self.dynamic_points is None else self.dynamic_points.copy()
            sta = None if self.static_points  is None else self.static_points.copy()
            dyn_header = self.dynamic_header
            sta_header = self.static_header

        markers = MarkerArray()

        # RViz残骸クリア
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        if dyn is not None and dyn.shape[0] > 0 and dyn_header is not None:
            d_markers = self.cluster_and_make_markers(
                dyn, DYNAMIC_COLOR, dyn_header.frame_id, dyn_header.stamp, ns="clusters"
            )
            markers.markers.extend(d_markers)

        if sta is not None and sta.shape[0] > 0 and sta_header is not None:
            s_markers = self.cluster_and_make_markers(
                sta, STATIC_COLOR, sta_header.frame_id, sta_header.stamp, ns="static", is_static=True
            )
            markers.markers.extend(s_markers)

        if len(markers.markers) == 1 and not self.publish_empty:
            return

        if len(markers.markers) > (self.max_clusters + 1):
            markers.markers = markers.markers[:(self.max_clusters + 1)]

        self.pub.publish(markers)

    # ---------- Clustering ----------
    def cluster_and_make_markers(self, points: np.ndarray, color: tuple, frame_id: str, stamp, ns: str, is_static: bool=False):
        points = self._apply_roi(points)
        if points.shape[0] == 0:
            return []

        pts_ds = voxel_downsample_numpy(points, self.voxel_size)
        if pts_ds.shape[0] == 0:
            return []

        # dynamicのみ：背景抑制（静的に近い差分点を消す）
        if not is_static:
            pts_ds = self._suppress_near_static(pts_ds, frame_id)
            if pts_ds.shape[0] == 0:
                return []

        # ノイズ除去（両方に効くが、特にdynamicに効く）
        pts_ds = self._radius_outlier_removal(pts_ds)
        if pts_ds.shape[0] == 0:
            return []

        pre_min = max(2, int(self.pre_min_cluster_size))
        clusters = []

        # region-growing
        N = pts_ds.shape[0]
        visited = np.zeros(N, dtype=bool)
        tree = cKDTree(pts_ds)

        for i in range(N):
            if visited[i]:
                continue

            stack = [i]
            visited[i] = True
            members = []

            while stack:
                u = stack.pop()
                members.append(u)

                z_u = float(pts_ds[u][self._logical_z_index()])
                eps_u = self._eps_for_forward(z_u)

                neigh = tree.query_ball_point(pts_ds[u], r=eps_u)
                for v in neigh:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)

            if len(members) < pre_min:
                continue

            cluster_pts = pts_ds[members]
            min_pt = cluster_pts.min(axis=0)
            max_pt = cluster_pts.max(axis=0)
            center = (min_pt + max_pt) / 2.0
            clusters.append({"min": min_pt, "max": max_pt, "center": center, "n": int(len(members))})

        if len(clusters) == 0:
            return []

        # dynamicのみ：マージ + NMS を強める
        if not is_static:
            clusters = self._merge_clusters_bbox(clusters)
            clusters = self._nms_clusters(clusters)

        # marker化
        markers = []
        marker_id = 0
        for c in clusters:
            center = c["center"]
            size = c["max"] - c["min"]

            if (not is_static):
                if not self._size_filter_ok(size):
                    continue
                if not self._quality_filter_ok(size):
                    continue
                req = self._required_cluster_size(self._forward_distance_of_center(center))
                if c["n"] < req:
                    continue

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
