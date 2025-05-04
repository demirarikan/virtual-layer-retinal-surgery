# Joint Repository for Papers: <br> Real-time Deformation-aware Control for Autonomous Robotic Subretinal Injection under iOCT Guidance (ICRA 2025) <br> & <br> Towards Motion Compensation in Autonomous Robotic Subretinal Injections (ISMR 2025)


**Real-time Deformation-aware Control for Autonomous Robotic Subretinal Injection under iOCT Guidance:** [Arxiv](https://arxiv.org/abs/2411.06557) 

**Towards Motion Compensation in Autonomous Robotic Subretinal Injections:** [Arxiv](https://arxiv.org/abs/2411.18521) 


## Overview 
This repository contains the code for a real-time deformation-aware control system designed for autonomous robotic subretinal injection. The system is guided by intraoperative Optical Coherence Tomography (iOCT) to ensure precision during delicate surgical procedures.

## Subretinal injection OCT segmentation dataset
Dataset used to train the segmentation network in these works can be found [here](https://github.com/demirarikan/subretinal-injection-oct-dataset).

## Features
- Real-time deformation-aware control
- Autonomous robotic subretinal injection
- iOCT guidance for enhanced precision
- Up and down eye motion compensation

## Installation
To use this repository, clone it and install the necessary dependencies:
- ROS
- MONAI
- PyTorch and torchvision
- Numpy
- Scipy
- Open3D

## Usage
>[!WARNING]
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

## 📄 Cite These Works

If you use this repository or find it helpful in your research, please cite:

[Real-time Deformation-aware Control for Autonomous Robotic Subretinal Injection under iOCT Guidance](https://arxiv.org/abs/2411.06557)

[Towards Motion Compensation in Autonomous Robotic Subretinal Injections](https://arxiv.org/abs/2411.18521)

BibTeX:
```bibtex
@article{Demir2024realtime,
  title={Real-time Deformation-aware Control for Autonomous Robotic Subretinal Injection under iOCT Guidance},
  author={Demir Arikan, Peiyao Zhang, Michael Sommersperger, Shervin Dehghani, Mojtaba Esfandiari, Russel H. Taylor, M. Ali Nasseri, Peter Gehlbach, Nassir Navab, Iulian Iordachita},
  journal={arXiv preprint arXiv:2411.06557},
  year={2024}
}

@article{Demir2024towards,
  title={Towards Motion Compensation in Autonomous Robotic Subretinal Injections},
  author={Demir Arikan, Peiyao Zhang, Michael Sommersperger, Shervin Dehghani, Mojtaba Esfandiari, Russel H. Taylor, M. Ali Nasseri, Peter Gehlbach, Nassir Navab, Iulian Iordachita},
  journal={arXiv preprint arXiv:2411.18521},
  year={2024}
}
```
