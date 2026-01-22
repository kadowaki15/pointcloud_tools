#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
import cv2
import numpy as np

class RGBObjectLocator(Node):
    def __init__(self):
        super().__init__('rgb_object_locator')

        # -------------------------
        # Topics
        # -------------------------
        self.color_topic = '/camera/color/image_raw'
        self.depth_topic = '/camera/depth/image_rect_raw'
        self.info_topic  = '/camera/depth/camera_info'
        self.out_topic   = '/rgb_detections_markers'

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter('min_area', 400)
        self.min_area = self.get_parameter('min_area').value

        self.bridge = CvBridge()
        self.prev_gray = None
        self.latest_depth = None

        # Camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # -------------------------
        # ROS I/O
        # -------------------------
        self.create_subscription(Image, self.color_topic, self.color_cb, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_cb, 10)
        self.create_subscription(CameraInfo, self.info_topic, self.info_cb, 10)

        self.pub = self.create_publisher(MarkerArray, self.out_topic, 10)

        self.get_logger().info("RGBObjectLocator started (metric output)")

    # -------------------------
    # Callbacks
    # -------------------------
    def info_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_cb(self, msg: Image):
        self.latest_depth = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough')

    def color_cb(self, msg: Image):
        if self.latest_depth is None or self.fx is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return

        # -------------------------
        # Motion detection
        # -------------------------
        diff = cv2.absdiff(self.prev_gray, gray)
        _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion = cv2.medianBlur(motion, 5)

        contours, _ = cv2.findContours(
            motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        markers = MarkerArray()
        mid = 0

        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue

            M = cv2.moments(c)
            if M['m00'] == 0:
                continue

            u = int(M['m10'] / M['m00'])
            v = int(M['m01'] / M['m00'])

            z = self.latest_depth[v, u] / 1000.0
            if z <= 0.1 or z > 5.0:
                continue

            # -------------------------
            # Pixel → camera metric
            # -------------------------
            x = (u - self.cx) * z / self.fx
            y = (v - self.cy) * z / self.fy

            m = Marker()
            m.header.frame_id = 'camera_depth_optical_frame'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'rgb'
            m.id = mid
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = float(z)

            m.scale.x = m.scale.y = m.scale.z = 0.25
            m.color.g = 1.0
            m.color.a = 0.8

            # 消え残り防止
            m.lifetime.sec = 0
            m.lifetime.nanosec = int(0.3 * 1e9)

            markers.markers.append(m)
            mid += 1

        if markers.markers:
            self.pub.publish(markers)

        self.prev_gray = gray


def main(args=None):
    rclpy.init(args=args)
    node = RGBObjectLocator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
