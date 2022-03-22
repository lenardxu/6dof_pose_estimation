# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
from os import path

_C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = _C

# set the limits on the state variables
_C.STATE_LIMIT = [10, 10, 10, 1]  # TODO the first three is dummy, which need to be checked from the dataset

# set the dim of image in terms of its height and width, which are exchanged for convenience -> (W, H)
_C.DIM = [640, 480]

# set the prob parameter of occlusion (binary) propagation: 1. prev:0 & curr: 0 -> 0.9; 2. prev:1 & curr: 0 -> 0.3;
_C.occl_prob = [0.9, 0.3]

# set the prior of not being occluded for initialization
# TODO values for testing: 0, 0.005, 0.01, 0.3, 0.9
_C.init_vis_prob = 0.3

# set the pending methods for resampling
_C.resample_methods = ["multinomial", "residual", "stratified", "systematic"]

# set the lambda value which indicates the half-life
_C.lambda_ = 1

# set the standard deviation of detection of object model
# TODO to check the reasonableness of this parameter value
_C.std_m = 0.01

# set the configuration parameters of camera
## maximum depth (unit:m) that can be measured by the depth camera
_C.m = 6
## the weight of tails of Gaussian Mixture Distribution for camera model
_C.beta = 0.01
## standard deviation of measurement of camera
## Ref.: Accuracy and Resolution of Kinect Depth Data for Indoor Mapping Applications (Fig. 10)
_C.std_c = 0.008

# set the distance threshold (unit:m) for e.g. comparing the pixel depth bound to the measured obj and pixel depth bound
# to the tracked obj
# TODO the dist_thres is not correctly set
_C.dist_thres = 0.05

# set the smoothing value
# TODO to check the reasonableness of this parameter value
_C.smooth_val = 1.e-6

# set the path to read the sequence of images and save results
_C.img_path = path.join(path.dirname(path.dirname(path.dirname(__file__))), "data", "0044")
_C.output_path = path.join(path.dirname(path.dirname(path.dirname(__file__))), "results", "0044")

# set the number indicating the object being tracked in the demo
_C.obj_id = 1

# set the standard deviation of initial position and position in the following steps which serve as reference for prediction
_C.std_pos_init = 0.05
_C.std_pos = 0.02

# set the number of images for testing
#TODO to set it as 800 afterwards
_C.test_no = 50


