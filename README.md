# Real-time Deformation-aware Control for Autonomous Robotic Subretinal Injection under iOCT Guidance

**Paper:** [Arxiv](https://arxiv.org/abs/2411.06557) 

## Overview 
This repository contains the code for a real-time deformation-aware control system designed for autonomous robotic subretinal injection. The system is guided by intraoperative Optical Coherence Tomography (iOCT) to ensure precision during delicate surgical procedures.

## Features
- Real-time deformation-aware control
- Autonomous robotic subretinal injection
- iOCT guidance for enhanced precision

## Installation
To use this repository, clone it and install the necessary dependencies:
- ROS
- MONAI
- PyTorch and torchvision
- Numpy
- Scipy
- Open3D

## Usage

>[INFO]
>Due to legal reasons we are not allowed to share the contents of the leica reader, which reads and converts Leica iOCT signals into numpy arrays.

### B<sup>5</sup>-scans
This video shows how to setup the Leica microscope for B<sup>5</sup>-scan acquisition.

https://github.com/user-attachments/assets/839f2aef-d5d9-4d78-9c4f-c91f6c62bcf6

### Setting up ROS nodes

- Start the b-scan publisher
```python
python3 b_scan_publisher.py
```
- Start the new robot controller
```python
python3 new_robot_controller.py
```
- Start the depth control script
  
>[!WARNING]
>This will start moving the robot! Make sure the emergency stop button is within reach.

Target depth and the maximum insertion velocity can be changed in the main function of this script as well. 
```python
python3 ros_depth_control.py
```
