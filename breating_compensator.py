from oct_point_cloud import OctPointCloud
import numpy as np

class BreathingCompensator():
    def __init__(self):
        self.curr_depth, self.prev_depth = None, None

    def calc_breath_motion(self, seg_volume):
        oct_pcd = OctPointCloud(seg_volume=seg_volume)
        print(oct_pcd.ilm_points[:10])
        avg_depth = np.median(oct_pcd.ilm_points, axis=0)[1]
        # return avg_depth
        self.prev_depth, self.curr_depth = self.curr_depth, avg_depth
        if self.curr_depth and self.prev_depth:
            pix_diff = self.prev_depth - self.curr_depth
            mm_diff = pix_diff * 3.379 / 1024
            print(f"prev:{self.prev_depth}\ncurr:{self.curr_depth}\ndiff:{pix_diff}pix / {mm_diff}mm")

            return mm_diff
        else:
            return 0
