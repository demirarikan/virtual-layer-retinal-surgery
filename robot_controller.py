import rospy
from geometry_msgs.msg import Vector3, Transform
from std_msgs.msg import Bool, Float64
import numpy as np
from scipy.spatial.transform import Rotation as R

import time


class RobotController:
    def __init__(self):
        # rospy.init_node("robot_controller_depth")
        self.robot_ee_frame_sub = rospy.Subscriber(
            "/eye_robot/FrameEE", Transform, self.update_pos_or
        )
        self.pub_tip_vel = rospy.Publisher(
            "/eyerobot2/desiredTipVelocities", Vector3, queue_size=3
        )
        self.pub_tip_vel_angular = rospy.Publisher(
            "/eyerobot2/desiredTipVelocitiesAngular", Vector3, queue_size=3
        )
        self.pub_cont_stop_sig = rospy.Publisher("stop_cont_pub", Bool, queue_size=3)
        self.pub_cont_vel = rospy.Publisher("cont_mov_vel", Float64, queue_size=3)
        self.pub_insertion_dir = rospy.Publisher(
            "cont_mov_dir", Bool, queue_size=0
        )  # True: Forward needle axis, False: Backward needle axis
        rospy.sleep(0.5)
        self.position = []
        self.orientation = []

    def update_pos_or(self, data):
        x = data.translation.x
        y = data.translation.y
        z = data.translation.z
        rx = data.rotation.x
        ry = data.rotation.y
        rz = data.rotation.z
        rw = data.rotation.w
        self.position = np.array([x, y, z])
        self.orientation = np.array([rx, ry, rz, rw])

    def move_forward_needle_axis(self, kp_linear_vel=2, linear_vel=0.1, duration_sec=1):
        current_quat = self.orientation
        r_current = R.from_quat(current_quat)
        rotation_matrix_current = r_current.as_matrix()
        moving_direction = np.matmul(rotation_matrix_current, np.array((0, 0, -1)))
        send_linear_velocity = moving_direction * kp_linear_vel * linear_vel

        for _ in range(int(duration_sec / 0.1)):
            self.pub_tip_vel.publish(
                send_linear_velocity[0],
                send_linear_velocity[1],
                send_linear_velocity[2],
            )
            rospy.sleep(0.1)
        self.pub_tip_vel.publish(0, 0, 0)
        rospy.sleep(0.1)

    def move_backward_needle_axis(
        self, kp_linear_vel=2, linear_vel=0.1, duration_sec=1
    ):
        current_quat = self.orientation
        r_current = R.from_quat(current_quat)
        rotation_matrix_current = r_current.as_matrix()
        moving_direction = np.matmul(rotation_matrix_current, np.array((0, 0, 1)))
        send_linear_velocity = moving_direction * kp_linear_vel * linear_vel

        for _ in range(int(duration_sec / 0.1)):
            self.pub_tip_vel.publish(
                send_linear_velocity[0],
                send_linear_velocity[1],
                send_linear_velocity[2],
            )
            rospy.sleep(0.1)
        self.pub_tip_vel.publish(0, 0, 0)

    def stop(self):
        for i in range(10):
            self.pub_tip_vel.publish(0, 0, 0)
            rospy.sleep(0.01)

    def start_cont_insertion(self):
        self.pub_cont_stop_sig.publish(False)

    def stop_cont_insertion(self):
        self.pub_cont_stop_sig.publish(True)

    def __calculate_robot_vel(self, current_depth, target_depth, method):
        threshold = target_depth * 0.1
        difference = abs(target_depth - current_depth)
        if difference < threshold:
            return 0        
        max_vel = 0.3
        if method == "linear":
            y_intercept = max_vel
            x_intercept = target_depth
            vel = min(
                max_vel, (-(y_intercept / x_intercept) * current_depth) + y_intercept
            )
            vel = max(vel, 0)

        
        elif method == "exponential":
            vel = min(difference**2, max_vel)

        if difference < threshold:
            vel = vel * 0.1        
        return vel

    def adjust_movement(
        self,
        current_depth_relative,
        target_depth_relative,
        error_range=0.05,
        method="linear",
    ):
        if current_depth_relative >= target_depth_relative + error_range:
            self.pub_insertion_dir.publish(False)
        else:
            self.pub_insertion_dir.publish(True)
            vel = self.__calculate_robot_vel(
                current_depth_relative, target_depth_relative, method
            )
            self.pub_cont_vel.publish(vel)


    def breath_motion(self, depth_diff):
        vel_gain = -0.08
        if depth_diff > 0:
            vel_gain = -vel_gain
        curr_pos = self.position
        target_pos = np.array((curr_pos[0], curr_pos[1], curr_pos[2] - depth_diff))
        # diff_norm = diff / np.linalg.norm(diff)
        # linear_vel = diff_norm * vel_gain
        print(abs(curr_pos[2] - target_pos[2]))
        if (abs(curr_pos[2] - target_pos[2]) > 0.001):
            print(f"sending tip vel: {vel_gain}")
            for i in range(2):
                self.pub_tip_vel.publish(0, 0, vel_gain)
        else:
            print("no motion")
            self.pub_tip_vel.publish(0,0,0)

# if __name__ == "__main__":
#     rospy.init_node("rob_cont_test")
#     cont = RobotController()
#     time.sleep(3)
#     cont.poke()
