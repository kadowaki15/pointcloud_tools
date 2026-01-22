#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
from collections import deque, Counter
import struct
import time
import math

# optional KDTree
try:
    from scipy.spatial import cKDTree as KDTree  # type: ignore
    HAVE_KDTREE = True
except Exception:
    HAVE_KDTREE = False

def pointcloud2_to_xyz_array(pc2_msg, max_points=None):
    fmt_float = 'f'
    offset = {}
    for fld in pc2_msg.fields:
        if fld.name in ('x', 'y', 'z'):
            offset[fld.name] = fld.offset
    if not all(k in offset for k in ('x', 'y', 'z')):
        offset = {'x': 0, 'y': 4, 'z': 8}

    step = pc2_msg.point_step
    data = pc2_msg.data
    n_points = int(len(data) / step) if step > 0 else 0
    if max_points is not None:
        n_points = min(n_points, max_points)
    pts = []
    for i in range(n_points):
        base = i * step
        try:
            x = struct.unpack_from(fmt_float, data, base + offset['x'])[0]
            y = struct.unpack_from(fmt_float, data, base + offset['y'])[0]
            z = struct.unpack_from(fmt_float, data, base + offset['z'])[0]
        except Exception:
            continue
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        pts.append((x, y, z))
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)

def euclidean_clusters(points, tol=0.08, min_size=10):
    N = points.shape[0]
    if N == 0:
        return []
    if HAVE_KDTREE:
        tree = KDTree(points)
        clusters = []
        visited = np.zeros(N, dtype=bool)
        for i in range(N):
            if visited[i]:
                continue
            idxs = [i]
            ptr = 0
            visited[i] = True
            while ptr < len(idxs):
                q = idxs[ptr]
                nbrs = tree.query_ball_point(points[q], tol)
                for nb in nbrs:
                    if not visited[nb]:
                        visited[nb] = True
                        idxs.append(nb)
                ptr += 1
            if len(idxs) >= min_size:
                clusters.append(np.array(idxs, dtype=int))
        return clusters
    else:
        dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        visited = np.zeros(N, dtype=bool)
        clusters = []
        for i in range(N):
            if visited[i]:
                continue
            idxs = [i]
            visited[i] = True
            ptr = 0
            while ptr < len(idxs):
                q = idxs[ptr]
                nbrs = np.where(dists[q] <= tol)[0]
                for nb in nbrs:
                    if not visited[nb]:
                        visited[nb] = True
                        idxs.append(int(nb))
                ptr += 1
            if len(idxs) >= min_size:
                clusters.append(np.array(idxs, dtype=int))
        return clusters

class StaticObstacleDetector(Node):
    def __init__(self):
        super().__init__('static_obstacle_detector')

        # パラメータ
        self.declare_parameter('input_topic', '/camera/camera/depth/color/points')
        self.declare_parameter('out_topic', '/static_obstacles')
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('ground_remove_y', -1.0)  # カメラ下方向除去（-1でOFF）
        self.declare_parameter('persistency_window', 2.0)
        self.declare_parameter('persistency_count_thresh', 3)
        self.declare_parameter('cluster_tol', 0.08)
        self.declare_parameter('min_points_per_cluster', 10)
        self.declare_parameter('publish_rate', 2.0)
        self.declare_parameter('frame_id', 'camera_depth_optical_frame')
        self.declare_parameter('max_points_per_frame', 100000)

        # load params
        self.input_topic = str(self.get_parameter('input_topic').value)
        self.out_topic = str(self.get_parameter('out_topic').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.ground_remove_y = float(self.get_parameter('ground_remove_y').value)
        self.persistency_window = float(self.get_parameter('persistency_window').value)
        self.persistency_count_thresh = int(self.get_parameter('persistency_count_thresh').value)
        self.cluster_tol = float(self.get_parameter('cluster_tol').value)
        self.min_points_per_cluster = int(self.get_parameter('min_points_per_cluster').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.max_points_per_frame = int(self.get_parameter('max_points_per_frame').value)

        # sub/pub
        self.sub = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.cb_pointcloud,
            qos_profile_sensor_data
        )
        self.pub = self.create_publisher(MarkerArray, self.out_topic, 10)

        # history structures
        self.frames = deque()
        self.voxel_counter = Counter()
        self.latest_voxel_centroid = {}

        # timer
        self.timer = self.create_timer(1.0 / max(0.1, self.publish_rate), self.on_timer)

        self.get_logger().info(f"StaticObstacleDetector started. in:{self.input_topic} out:{self.out_topic} "
                               f"voxel={self.voxel_size} persist_window={self.persistency_window}s")

    def cb_pointcloud(self, msg: PointCloud2):
        pts = pointcloud2_to_xyz_array(msg, max_points=self.max_points_per_frame)
        self.get_logger().debug(f"PC2 callback raw_count={len(pts)}")
        if pts.shape[0] == 0:
            return

        # 床除去（カメラ下方向を除去）
        if self.ground_remove_y >= 0.0:
            mask = pts[:, 1] > self.ground_remove_y
            pts = pts[mask]

        if pts.shape[0] == 0:
            return

        # voxelization
        inv_vs = 1.0 / self.voxel_size
        idxs = np.floor(pts[:, :3] * inv_vs).astype(np.int32)
        voxel_map = {}
        for i, vid in enumerate(map(tuple, idxs)):
            if vid not in voxel_map:
                voxel_map[vid] = []
            voxel_map[vid].append(i)

        voxel_centroids = {}
        for vid, indices in voxel_map.items():
            pts_sel = pts[indices, :3]
            centroid = tuple(np.mean(pts_sel, axis=0).tolist())
            voxel_centroids[vid] = centroid

        ts = time.time()
        self.frames.append((ts, voxel_centroids))
        for vid, cent in voxel_centroids.items():
            self.voxel_counter[vid] += 1
            self.latest_voxel_centroid[vid] = cent

        # drop old frames
        cutoff = ts - self.persistency_window
        while self.frames and self.frames[0][0] < cutoff:
            old_ts, old_map = self.frames.popleft()
            for vid in old_map.keys():
                self.voxel_counter[vid] -= 1
                if self.voxel_counter[vid] <= 0:
                    del self.voxel_counter[vid]
                    if vid in self.latest_voxel_centroid:
                        del self.latest_voxel_centroid[vid]

        self._last_pc_frame_id = msg.header.frame_id if hasattr(msg, 'header') else self.frame_id

    def on_timer(self):
        persistent_voxels = [vid for vid, cnt in self.voxel_counter.items()
                             if cnt >= self.persistency_count_thresh]

        if len(persistent_voxels) == 0:
            return

        pts = []
        for vid in persistent_voxels:
            if vid in self.latest_voxel_centroid:
                pts.append(self.latest_voxel_centroid[vid])
        if len(pts) == 0:
            return
        pts_np = np.array(pts, dtype=np.float32)

        clusters_idx = euclidean_clusters(pts_np, tol=self.cluster_tol, min_size=self.min_points_per_cluster)

        # Build MarkerArray
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        pub_frame = getattr(self, '_last_pc_frame_id', self.frame_id)
        for cid, inds in enumerate(clusters_idx):
            cluster_pts = pts_np[inds]
            centroid = cluster_pts.mean(axis=0)
            mins = cluster_pts.min(axis=0)
            maxs = cluster_pts.max(axis=0)
            size = (maxs - mins)
            min_size = 0.05
            size = np.maximum(size, min_size)

            m = Marker()
            m.header.frame_id = pub_frame
            m.header.stamp = stamp
            m.ns = 'static'
            m.id = int(cid)
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(centroid[0])
            m.pose.position.y = float(centroid[1])
            m.pose.position.z = float(centroid[2])
            m.pose.orientation.w = 1.0
            m.scale.x = float(size[0])
            m.scale.y = float(size[1])
            m.scale.z = float(size[2])
            m.color.r = 1.0
            m.color.g = 0.6
            m.color.b = 0.0
            m.color.a = 0.6
            m.lifetime.sec = 2
            markers.markers.append(m)

            t = Marker()
            t.header = m.header
            t.ns = 'static'
            t.id = 10000 + int(cid)
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(centroid[0])
            t.pose.position.y = float(centroid[1])
            t.pose.position.z = float(centroid[2]) + max(size[2], 0.2) + 0.1
            t.scale.z = 0.18
            t.color.r = 1.0
            t.color.g = 1.0
            t.color.b = 1.0
            t.color.a = 1.0
            t.text = f"pts={len(inds)}"
            t.lifetime.sec = 2
            markers.markers.append(t)

        if len(markers.markers) > 0:
            self.pub.publish(markers)
            self.get_logger().info(f"Published {len(clusters_idx)} static clusters (markers={len(markers.markers)})")

def main(args=None):
    rclpy.init(args=args)
    node = StaticObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
