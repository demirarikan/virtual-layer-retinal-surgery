import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Float32, Bool
from geometry_msgs.msg import Vector3, Transform

from oct_point_cloud import OctPointCloud
from needle_seg_model import NeedleSegModel
from image_conversion_without_using_ros import image_to_numpy

import numpy as np



class BreathSim():
    def __init__(self, seg_model):
        # self.b_scan_sub = rospy.Subscriber("oct_b_scan", Image, self.b_scan_callback, queue_size=3)
        self.robot_ee_frame_sub = rospy.Subscriber(
            "/eye_robot/FrameEE", Transform, self.update_robot_pos
        )
        self.pub_tip_vel = rospy.Publisher(
            "/eyerobot2/desiredTipVelocities", Vector3, queue_size=3
        )
        self.layer_depth_sub = rospy.Subscriber("layer_depth", Float32, self.update_layer_depth)
        
        self.cv_bridge = CvBridge()
        self.latest_b5_vol = []
        self.position = []
        self.orientation = []

        self.seg_model = seg_model

        self.prev_depth, self.curr_depth = None, None

    
    def update_robot_pos(self, data):
        x = data.translation.x
        y = data.translation.y
        z = data.translation.z
        rx = data.rotation.x
        ry = data.rotation.y
        rz = data.rotation.z
        rw = data.rotation.w
        self.position = np.array([x, y, z])
        self.orientation = np.array([rx, ry, rz, rw])


    # def b_scan_callback(self, data):
    #     b_scan = image_to_numpy(data)
    #     self.latest_b5_vol.append(b_scan)
    #     if len(self.latest_b5_vol) == 5:
    #         np_b5_vol = np.array(self.latest_b5_vol)
    #         seg_volume = self.segment_volume(np_b5_vol)
    #         layer_depth = self.calc_layer_depth(seg_volume)
    #         self.prev_depth, self.curr_depth = self.curr_depth, layer_depth
    #         print(self.prev_depth, self.curr_depth)
    #         if self.curr_depth and self.prev_depth:
    #             pix_diff = self.prev_depth - self.curr_depth
    #             mm_diff = pix_diff * 3.379 / 1024
    #             print(f"prev:{self.prev_depth}\ncurr:{self.curr_depth}\ndiff:{pix_diff}pix / {mm_diff}mm")

    #             self.robot_breathing_motion(depth_diff=mm_diff)
    #         else:
    #             self.robot_breathing_motion(depth_diff=0)
    #         self.latest_b5_vol = []

    # def segment_volume(self, oct_volume):
    #     oct_volume = self.seg_model.preprocess_volume(oct_volume)
    #     seg_volume = self.seg_model.segment_volume(oct_volume)
    #     seg_volume = self.seg_model.postprocess_volume(seg_volume)
    #     return seg_volume

    # def calc_layer_depth(self, seg_volume):
    #     oct_pcd = OctPointCloud(seg_volume)
    #     avg_depth = np.median(oct_pcd.ilm_points, axis=0)[1]
    #     return avg_depth

    def update_layer_depth(self, depth_data):
        layer_depth = depth_data.data
        self.prev_depth, self.curr_depth = self.curr_depth, layer_depth
        print(self.prev_depth, self.curr_depth)
        if self.curr_depth and self.prev_depth:
            pix_diff = self.prev_depth - self.curr_depth
            mm_diff = pix_diff * 3.379 / 1024
            print(f"prev:{self.prev_depth}\ncurr:{self.curr_depth}\ndiff:{pix_diff}pix / {mm_diff}mm")

            self.robot_breathing_motion(depth_diff=mm_diff)
        else:
            self.robot_breathing_motion(depth_diff=0)

    def robot_breathing_motion(self, depth_diff):
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

if __name__ == "__main__":
    rospy.init_node("breath_sim")
    # seg_model = NeedleSegModel(None, "weights/best_150_val_loss_0.4428_in_retina.pth")
    depth_control = BreathSim(None)
    rospy.spin()