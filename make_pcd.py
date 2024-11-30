import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import numpy as np
import struct


def pointcloud_callback(msg):
    # Parse the PointCloud2 message
    points_gen = pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)

    # Prepare lists for points and colors
    points = []
    colors = []

    # Iterate through the points and extract RGB values
    for point in points_gen:
        x, y, z, rgb = point

        # Unpack the float into an integer and extract color channels
        packed = struct.unpack('I', struct.pack('f', rgb))[0]
        r = ((packed >> 16) & 0xFF) / 255.0  # Normalize to [0, 1]
        g = ((packed >> 8) & 0xFF) / 255.0
        b = (packed & 0xFF) / 255.0

        # Append to lists
        points.append([x, y, z])
        colors.append([r, g, b])

    # Convert lists to numpy arrays
    points = np.asarray(points)
    colors = np.asarray(colors)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    visualize_point_cloud(pcd)

def visualize_point_cloud(pcd):
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud", width=800, height=600, left=50, top=50)

def save_pointcloud():
    rospy.init_node("pointcloud_saver", anonymous=True)
    rospy.Subscriber("azure_kinect/points2", PointCloud2, pointcloud_callback)
    rospy.loginfo("Subscribed to '/points2' topic. Waiting for PointCloud2 messages...")
    rospy.spin()

if __name__ == "__main__":
    save_pointcloud()
