import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time

import mock_components
from depth_calculator import DepthCalculator
from logger import Logger
from needle_seg_model import NeedleSegModel

def process_latest_scan(
    leica_reader, seg_model, depth_calculator, robot_controller, logger, target_depth_relative
):

    oct_volumes = {}
    seg_volumes = {}
    depths = {}
    pcd = {}
    count = 0  # image count

    error_range = 0.05
    print("Processing latest scan")
    while True:
        try:
            scan = leica_reader.fast_get_b_scan_volume()
        except StopIteration:
            break

        start_time = time.perf_counter()
        oct_volume = seg_model.preprocess_volume(scan)
        oct_volumes[count] = oct_volume
        seg_volume = seg_model.segment_volume(oct_volume)
        # segmentation result is converted to type uint8 for faster processing in the next steps!!!
        seg_volume = seg_model.postprocess_volume(seg_volume)
        seg_volumes[count] = seg_volume

        current_depth_relative, geo_components = depth_calculator.calculate_depth(
            seg_volume, log_final_pcd=True
        )

        print(f'Current relative depth: {current_depth_relative}.')

        depths[count] = current_depth_relative
        pcd[count] = geo_components

        robot_controller.adjust_movement(current_depth_relative, target_depth_relative)
        print(f'duration: {time.perf_counter() - start_time}')
        count += 1

        if (
            (current_depth_relative >= 0
            and abs(current_depth_relative - target_depth_relative) < error_range)
            or current_depth_relative > target_depth_relative
        ):
            robot_controller.stop()
            print(f"Stopping robot at depth {current_depth_relative}")
            # logger.save_logs(oct_volumes, seg_volumes, pcd, depths)
            break
    logger.save_logs(oct_volumes, seg_volumes, pcd, depths)

if __name__ == "__main__":
    leica_reader = mock_components.LeicaEngineMock(
            "/home/demir/Desktop/jhu_project/oct_scans/jul30/2-depth-cont/oct_volumes"
        )
    robot_controller = mock_components.RobotControllerMock()
    seg_model = NeedleSegModel(None, "weights/best_150_val_loss_0.4428_in_retina.pth")
    logger = Logger()
    depth_calculator = DepthCalculator(None)
    process_latest_scan(
        leica_reader,
        seg_model,
        depth_calculator,
        robot_controller,
        logger,
        0.5
    )
    
