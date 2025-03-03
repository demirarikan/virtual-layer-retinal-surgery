import socket
import struct

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rospy
from geometry_msgs.msg import Transform


class LeicaEngine(object):

    def __init__(self,):
        raise NotImplementedError
    
    def fast_get_b_scan_volume(self):
        raise NotImplementedError
    
    def __get_b_scans_volume__(self):
        raise NotImplementedError

    def __calculate_spacing(self):
        raise NotImplementedError
    
    def __disconnect__(self):
        raise NotImplementedError

    def __connect__(self):
        raise NotImplementedError

    def __get_buffer__(self):
        raise NotImplementedError

    def __parse_data__(self):
        raise NotImplementedError

    def get_b_scan(self):  # frame_to_save(0 for upper, 1 for lower)
        raise NotImplementedError
