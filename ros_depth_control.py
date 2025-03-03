import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Float32, Bool
from oct_point_cloud import OctPointCloud
from needle_seg_model import NeedleSegModel
from logger import Logger
import numpy as np
from image_conversion_without_using_ros import image_to_numpy
import time


class ROSDepthControl:
    def __init__(self, target_depth, max_vel, breathing_compensation, seg_model, logger):
        # ROS components
        self.insertion_vel_pub = rospy.Publisher("cont_mov_vel", Float64, queue_size=1)
        self.insertion_stop_pub = rospy.Publisher("stop_cont_pub", Bool, queue_size=1)
        self.b_scan_sub = rospy.Subscriber("oct_b_scan", Image, self.b_scan_callback, queue_size=3)
        # for breathing compensation
        self.layer_depth_pub = rospy.Publisher("layer_depth", Float32, queue_size=1)

        self.latest_b5_vol = []

        # insertion parameters
        self.target_depth = target_depth
        self.max_vel = max_vel
        self.insertion_complete = False
        self.breathing_compensation = breathing_compensation

        # components
        self.seg_model = seg_model
        self.logger = logger


        self.log = {
            "b5_scans": [],             # b5 scan volume
            "segmented_volumes": [],    # segmented volume
            "needle_tip_coords": [],    # needle tip coordinates
            "needle_tip_depth": [],     # needle tip depth as percentage of retina thickness
            "ilm_rpe_z_coords": [],     # ilm and rpe z coordinates at needle tip a-scan
            "insertion_velocity": [],   # insertion velocity calculated at each step
            "avg_layer_depth": [],      # average z pos of ilm points for breathing compensation
        }

    def b_scan_callback(self, data):
        b_scan = image_to_numpy(data)
        self.latest_b5_vol.append(b_scan)
        if len(self.latest_b5_vol) == 5:
            print("Started B5-scan processing")
            # start_time = time.perf_counter()
            np_b5_vol = np.array(self.latest_b5_vol)
            seg_vol = self.segment_volume(np_b5_vol)
            needle_tip_coords, inpainted_ilm, inpainted_rpe = self.process_pcd(seg_vol)

            if not self.insertion_complete:

                self.log["b5_scans"].append(np_b5_vol)
                self.log["segmented_volumes"].append(seg_vol)
                self.log["needle_tip_coords"].append(needle_tip_coords)

                _, _, needle_depth, ilm_rpe_z_coords = self.calculate_needle_depth(
                    needle_tip_coords, inpainted_ilm, inpainted_rpe
                )
                print(f"Estimated needle depth: {needle_depth}")

                self.log["needle_tip_depth"].append(needle_depth)
                self.log["ilm_rpe_z_coords"].append(ilm_rpe_z_coords)

                insertion_vel = self.update_insertion_velocity(needle_depth)

                self.log["insertion_velocity"].append(insertion_vel)

            self.latest_b5_vol = []
            if self.insertion_complete and not self.breathing_compensation:
                rospy.signal_shutdown("Insertion complete, no breathing compensation. Shutting down...")    
            # print(f"Took: {time.perf_counter()-start_time} seconds")

    def segment_volume(self, oct_volume):
        oct_volume = self.seg_model.preprocess_volume(oct_volume)
        seg_volume = self.seg_model.segment_volume(oct_volume)
        seg_volume = self.seg_model.postprocess_volume(seg_volume)
        return seg_volume

    def process_pcd(self, seg_volume):
        oct_pcd = OctPointCloud(seg_volume)
        # ROS depth publisher for breathing compensation
        avg_depth = np.median(oct_pcd.ilm_points, axis=0)[1]
        # only log avg ilm layer depth if breathing compensation is done
        if self.breathing_compensation:
            self.log["avg_layer_depth"].append(avg_depth)
        self.layer_depth_pub.publish(avg_depth)

        needle_tip_coords = oct_pcd.find_needle_tip()
        inpainted_ilm, inpainted_rpe = oct_pcd.inpaint_layers()
        return needle_tip_coords, inpainted_ilm, inpainted_rpe

    def calculate_needle_depth(self, needle_tip_coords, inpainted_ilm, inpainted_rpe):
        needle_tip_depth = needle_tip_coords[1]
        ilm_depth = inpainted_ilm[needle_tip_coords[0], needle_tip_coords[2]]
        rpe_depth = inpainted_rpe[needle_tip_coords[0], needle_tip_coords[2]]
        ilm_rpe_distance = rpe_depth - ilm_depth
        needle_tip_depth_relative = needle_tip_depth - ilm_depth
        needle_tip_depth_relative_percentage = (
            needle_tip_depth_relative / ilm_rpe_distance
        )
        return (
            needle_tip_depth,
            needle_tip_depth_relative,
            needle_tip_depth_relative_percentage,
            (int(ilm_depth), int(rpe_depth)),
        )

    def update_insertion_velocity(self, current_depth):
        insertion_vel = self.__calculate_insertion_velocity(current_depth)
        print(f"Setting velocity to: {insertion_vel}")
        if insertion_vel == 0:
            self.insertion_vel_pub.publish(0)
            self.insertion_stop_pub.publish(True)
            self.insertion_complete = True
        else:
            self.insertion_vel_pub.publish(insertion_vel)
        return insertion_vel


    def __calculate_insertion_velocity(self, current_depth, method="linear"):
        threshold = self.target_depth * 0.1
        difference = abs(self.target_depth - current_depth)
        # Stop the insertion if the needle is within the threshold
        if difference < threshold:
            return 0
        # Move needle back if it overshoots the target depth
        if current_depth > self.target_depth + threshold:
            return -0.2
        if method == "linear":
            y_intercept = self.max_vel
            x_intercept = self.target_depth
            vel = min(
                self.max_vel, (-(y_intercept / x_intercept) * current_depth) + y_intercept
            )
            vel = max(vel, 0)
        elif method == "exponential":
            vel = min(difference**2, self.max_vel)
        return vel
    
    def log_results(self):
        print("Logging results")
        self.logger.save_b5_scans(self.log["b5_scans"])
        self.logger.save_segmented_volumes_and_result_oct(self.log["b5_scans"], self.log["segmented_volumes"], self.log["needle_tip_coords"])
        self.logger.save_csv(self.log["needle_tip_depth"], self.log["ilm_rpe_z_coords"], self.log["insertion_velocity"], self.log["avg_layer_depth"])
        self.logger.save_pcd(self.log["segmented_volumes"], self.log["needle_tip_coords"])
        print("Done logging results")


if __name__ == "__main__":
    rospy.init_node("depth_control")

    def shutdown_hook():
        depth_control.log_results()
        print("Shutting down depth control node")

    rospy.on_shutdown(shutdown_hook)

    target_depth = 0.4
    max_vel = 0.4
    breathing_compensation = False
    seg_model = NeedleSegModel(None, "weights/best_150_val_loss_0.4428_in_retina.pth")
    logger = Logger(log_dir="/media/peiyao/SSD1T/demir/oct22")
    depth_control = ROSDepthControl(target_depth, max_vel, breathing_compensation, seg_model, logger)
    rospy.spin()
    # while not rospy.is_shutdown():
    #     continue