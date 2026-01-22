import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/kadowaki/realsense_ws/src/pointcloud_tools/install/pointcloud_tools'
