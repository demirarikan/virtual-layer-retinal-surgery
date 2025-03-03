#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Vector3, Transform
from std_msgs.msg import Bool, Float64
import numpy as np
from scipy.spatial.transform import Rotation as R

import time


class RobotController:
    def __init__(self):
        # insertion information
        self.stop_event_sub = rospy.Subscriber(
            "stop_cont_pub", Bool, self.update_stop_signal
        )
        self.cont_vel_sub = rospy.Subscriber(
            "cont_mov_vel", Float64, self.update_cont_vel
        )

        # robot information
        self.robot_ee_frame_sub = rospy.Subscriber(
            "/eye_robot/FrameEE", Transform, self.update_pos_or
        )
        self.pub_tip_vel = rospy.Publisher(
            "/eyerobot2/desiredTipVelocities", Vector3, queue_size=1
        )
        self.pub_tip_vel_angular = rospy.Publisher(
            "/eyerobot2/desiredTipVelocitiesAngular", Vector3, queue_size=1
        )
        # sleep here to give time for the publishers/subscirbers to initialize
        rospy.sleep(1)
        self.stop_signal = False
        self.position = []
        self.orientation = []

        # insertion params
        self.linear_vel = 0.0
        self.mov_dir = False

    def update_cont_vel(self, data):
        print(f"Linear velocity changed to: {data.data}")
        self.linear_vel = data.data
        if self.stop_signal == True:
            print("Velocity adjusted setting stop signal to false!")
            self.stop_signal = False

    def update_stop_signal(self, data):
        if data.data == False:
            print("Starting insertion")
        else:
            print("Stopping insertion")
        self.stop_signal = data.data

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

    def move_along_needle_axis(self, linear_vel=0.1):
        current_quat = self.orientation
        r_current = R.from_quat(current_quat)
        rotation_matrix_current = r_current.as_matrix()
        moving_direction = np.matmul(rotation_matrix_current, np.array((0, 0, -1)))
        send_linear_velocity = moving_direction * linear_vel
        self.pub_tip_vel.publish(
            send_linear_velocity[0],
            send_linear_velocity[1],
            send_linear_velocity[2],
        )

    def stop(self):
        # send stop signal in loop to make sure it stops
        # sometimes commands are lost if they are sent too quickly
        for i in range(5):
            self.pub_tip_vel.publish(0, 0, 0)

if __name__ == "__main__":
    rospy.init_node("cont_pub_node", disable_signals=True)
    rob_cont = RobotController()
    
    def shutdown_hook():
        # rob_cont.move_along_needle_axis(linear_vel=0) 
        rob_cont.stop()                              
        print("Shutting down continuous insertion controller")
    rospy.on_shutdown(shutdown_hook)
    
    print("started robot controller")
    time.sleep(0.5)

    while not rospy.is_shutdown():
        if not rob_cont.stop_signal:
            rob_cont.move_along_needle_axis(linear_vel=rob_cont.linear_vel)
            time.sleep(0.01)
        else:
            rob_cont.stop()
            # print("robot controller shut down")
            # rospy.signal_shutdown("Stop signal received")
            

