# Find where the robot is
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped

def callback(msg):
    # Get the current position
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    orientation = msg.pose.pose.orientation

    rospy.loginfo(f"Current position: x={x}, y={y}")
    rospy.loginfo(f"Orientation: {orientation}")

if __name__ == '__main__':
    rospy.init_node('get_robot_position_node')

    # Subscribe to the /amcl_pose topic
    rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, callback)

    rospy.spin()