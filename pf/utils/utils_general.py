# Ref Source:
# https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
import numpy as np
from numpy.random import randn, random, uniform, multivariate_normal, seed
import scipy.io as scio
from scipy.spatial.transform import Rotation as R
from os import path
from PIL import Image

from pf.utils.config import cfg


def multinomial_resample(weights):
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # avoid round-off errors
    return np.searchsorted(cumulative_sum, random(len(weights)))

def residual_resample(weights):
    N = len(weights)
    indexes = np.zeros(N, 'i')

    # take int(N*w) copies of each weight
    num_copies = (N*np.asarray(weights)).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]): # make n copies
            indexes[k] = i
            k += 1

    # use multinormial resample on the residual to fill up the rest.
    # TODO to check what w means
    residual = w - num_copies     # get fractional part
    residual /= sum(residual)     # normalize
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1. # ensures sum is exactly one
    indexes[k:N] = np.searchsorted(cumulative_sum, random(N-k))

    return indexes

def stratified_resample(weights):
    N = len(weights)
    # make N subdivisions, chose a random position within each one
    positions = (random(N) + range(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def systematic_resample(weights):
    N = len(weights)

    # make N subdivisions, choose positions
    # with a consistent random offset
    positions = (np.arange(N) + random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill(1.0 / len(weights))

def extract_pose_from_mat(img_path):
    meta_data = scio.loadmat(img_path)
    poses = meta_data['poses']
    # select the first object only for demo
    r_t_mat = poses[:, :, cfg.obj_id]  # (3,4)
    r = R.from_matrix(r_t_mat[:3, :3])  # Rotation objects
    r_quat_vec = r.as_quat()  # (4,)
    t_vec = r_t_mat[:3, 3]  # (3,)
    return np.concatenate((t_vec, r_quat_vec))

def extract_bboxes_from_dataset(meta_path, depth_img_path, bbox_path):
    """generate the dummy bboxes to replace the bboxes predicted by object detector"""
    meta_data = scio.loadmat(meta_path)
    cam_scale = meta_data['factor_depth'][0][0]
    depth_map = np.array(Image.open(depth_img_path)) / float(cam_scale)

    with open(bbox_path, 'r') as f:
        lines = f.readlines()
    bboxes_lst = []
    for line in lines:
        bbox = list(map(float, line.rstrip().split(" ")[1:]))
        for i in range(len(bbox)):
            bbox[i] = round(bbox[i])
        bbox[2] -= bbox[0]
        bbox[3] -= bbox[1]
        roi_depth_arr = depth_map[int(bbox[0]):int(bbox[0]+bbox[2]+1), int(bbox[1]):int(bbox[1]+bbox[3]+1)]
        temp = (tuple(bbox), roi_depth_arr)
        bboxes_lst.append(temp)
    bboxes = tuple(bboxes_lst)
    return bboxes



