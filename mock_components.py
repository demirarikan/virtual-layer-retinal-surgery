import os
import re
import time

import numpy as np


class LeicaEngineMock():
    def __init__(self, scans_path):
        self.scans = self.get_scans_iterator(scans_path)
        self.current_index = 0
        print("Mock Leica initialized")

    def get_scans_iterator(self, scans_path):

        _nsre = re.compile("([0-9]+)")

        def natural_sort_key(s):
            return [
                int(text) if text.isdigit() else text.lower()
                for text in re.split(_nsre, s)
            ]

        scans = [
            os.path.join(scans_path, scan)
            for scan in os.listdir(scans_path)
            if scan.endswith("npy") #scan.startswith("volume_") and 
        ]
        scans.sort(key=natural_sort_key)
        return scans

    def __get_b_scans_volume__(self):
        time.sleep(0.22)
        if self.current_index < len(self.scans):
            scan = self.scans[self.current_index]
            self.current_index += 1
            return np.load(scan)
        else:
            raise StopIteration("No more scans available.")
        
    def fast_get_b_scan_volume(self):
        time.sleep(0.3)
        if self.current_index < len(self.scans):
            scan = self.scans[self.current_index]
            self.current_index += 1
            return np.load(scan)
        else:
            raise StopIteration("No more scans available.")

class RobotControllerMock():
    def __init__(self):
        print("Mock Robot Controller initialized")

    def adjust_movement(self, current_depth_relative, target_depth_relative):
        print(f"Adjusting movement to {current_depth_relative} relative to {target_depth_relative}")

    def stop(self):
        print("Stopping robot")
