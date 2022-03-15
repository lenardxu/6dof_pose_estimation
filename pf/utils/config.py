# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

_C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = _C

# set the limits on the state variables
_C.STATE_LIMIT = [10, 10, 10, 1]  # TODO the first three is dummy, which need to be checked from the dataset

# set the dim of image in terms of its height and width
_C.DIM = [480, 640]

# set the prob parameter of occlusion (binary) propagation: 1. prev:0 & curr: 0 -> 0.9; 2. prev:1 & curr: 0 -> 0.3;
_C.occl_prob = [0.9, 0.3]

# set the prior of not being occluded for initialization
# TODO values for testing: 0, 0.005, 0.01, 0.3, 0.9
_C.init_vis_prob = 0.3

# set the pending methods for resampling
_C.resample_methods = ["multinomial", "residual", "stratified", "systematic"]