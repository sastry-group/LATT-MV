from mujoco_env_only_kuka import KukaTennisEnv
from stable_baselines3 import PPO
from geometry_msgs.msg import PoseStamped
import rospy
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf
from copy import deepcopy
import time
from scipy.spatial.transform import Rotation as R
import numpy as np
import argparse
from mujoco_env_only_kuka_ik import KukaTennisEnv as KukaTennisEnvIK
import moveit_commander
import copy
from mujoco_env_only_kuka_ik import KukaTennisEnv as KukaTennisEnvIK
from stable_baselines3 import PPO
from mujoco_env_kuka_with_table import KukaTennisEnv
# Import Twist
from geometry_msgs.msg import Twist

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='Enable rendering')

args = parser.parse_args()


RATE = 60
current_positions = None
angle_offset = 0.#np.pi/4.

# Callback function for the subscriber
def joint_state_publish(joint_angles,joint_velocities,joint_names):
    data = JointTrajectoryControllerState()
    # Extract the current joint positions and velocities from the state message
    data.actual.positions[:7] = joint_angles
    data.actual.velocities[:7] = joint_velocities
    data.joint_names = joint_names
    
def restrict_range(val, min_val, max_val):
    return min(max(val, min_val), max_val)

def command_callback(msg: JointTrajectory):
    global target_positions

    # Extract the current joint positions and velocities from the command message
    target_positions = np.array(msg.points[0].positions[:7])
    target_positions[0] = restrict_range(target_positions[0],-2.96,2.96)
    target_positions[1] = restrict_range(target_positions[1],-2.09,2.09)
    target_positions[2] = restrict_range(target_positions[2],-2.94,2.94)
    target_positions[3] = restrict_range(target_positions[3],-2.09,2.09)
    target_positions[4] = restrict_range(target_positions[4],-2.94,2.94)
    target_positions[5] = restrict_range(target_positions[5],-2.09,2.09)
    target_positions[6] = restrict_range(target_positions[6],-3,3)

def step():
    global env, target_positions
    
    current_positions = env.data.qpos[:7]
    current_velocities = env.data.qvel[:7]
    joint_names = ['lbr_iiwa_joint_1', 'lbr_iiwa_joint_2', 'lbr_iiwa_joint_3', 'lbr_iiwa_joint_4', 'lbr_iiwa_joint_5', 'lbr_iiwa_joint_6', 'lbr_iiwa_joint_7']
    joint_state_publish(current_positions, current_velocities, joint_names)
    
    ball_position = env.data.qpos[-7:-4]
    ball_velocity = env.data.qvel[-6:-3]
    ball_msg = Twist()
    ball_msg.linear.x = ball_position[0]
    ball_msg.linear.y = ball_position[1]
    ball_msg.linear.z = ball_position[2]
    ball_msg.angular.x = ball_velocity[0]
    ball_msg.angular.y = ball_velocity[1]
    ball_msg.angular.z = ball_velocity[2]
    ball_pub.publish(ball_msg)

    action = np.zeros(7)
    action = np.array(target_positions) - np.array(current_positions)
    _,_,done,_,_ = env.step(action)

    if done:
        obs, _ = env.reset()
    if args.render:
        env.render()
    
def Rot_z(z_deg) :
    z = np.radians(z_deg)
    return np.array([[np.cos(z), -np.sin(z), 0],
                    [np.sin(z), np.cos(z), 0],
                    [0, 0, 1]])

def Rot_x(x_deg) :
    x = np.radians(x_deg)
    return np.array([[1, 0, 0],
                    [0, np.cos(x), -np.sin(x)],
                    [0, np.sin(x), np.cos(x)]])


if __name__ == '__main__':
    global target_positions
    try:
        # Initialize the ROS node
        rospy.init_node('kuka_joint_mujoco_sim', anonymous=True)
        env = KukaTennisEnv(proc_id=1)
        obs, _ = env.reset()
        
        target_positions = np.zeros(7)

        # Publish to the joint state topic
        state_pub = rospy.Publisher('/lbr/PositionJointInterface_trajectory_controller/state', 
                         JointTrajectoryControllerState, queue_size=1)

        # Publish ball position and velocity to a Twist topic
        ball_pub = rospy.Publisher('/ball_info', Twist, queue_size=1)
        
        # Publisher for the joint command topic
        cmd_sub = rospy.Subscriber('/lbr/PositionJointInterface_trajectory_controller/command', 
                                         JointTrajectory, command_callback)

        # Keep the node alive and processing callbacks
        rate = rospy.Rate(RATE)  # 10 Hz
        while not rospy.is_shutdown():
            step()
            rate.sleep()

        

    except rospy.ROSInterruptException:
        pass

