from config import cfg
import numpy as np
from numpy.random import uniform
from scipy import stats


def occlusion_prob(prev_occl_state, curr_occl_state):
    """
    compute the occlusion propagation prob distribution per pixel
    :param prev_occl_state: int occlusion state of previous step
    :param curr_occl_state: int occlusion state of current step
    :return: prob distribution
    """
    if prev_occl_state == 0 & curr_occl_state == 0:
        return cfg.occl_prob[0]
    elif prev_occl_state == 0 & curr_occl_state == 1:
        return 1 - cfg.occl_prob[0]
    elif prev_occl_state == 1 & curr_occl_state == 0:
        return cfg.occl_prob[1]
    else:
        return 1 - cfg.occl_prob[1]

def occlusion_update(prev_occl_state, thres):
    """
    predict the current occlusion state based on current one according to given prob distribution per pixel
    :param prev_occl_state: int occlusion state of previous step
    :param thres: list prob parameters indicating occlusion (binary) propagation
    :return: int current occlusion state
    """
    sample = uniform(0, 1, size=1)
    if prev_occl_state == 0:
        if sample <= thres[0]:
            return 0
        else:
            return 1
    else:
        if sample <= thres[1]:
            return 0
        else:
            return 1

def likelihood(particle, bboxes, std_m, pixel_pos, depth_map, occluded):
    """

    :param particle: numpy.ndarray particles of shape (N, 7)
    :param bboxes: numpy.ndarray predicted bounding boxes of one row: [x_0, y_0, z_0, delta_x, delta_y, delta_z]
    :param std_m:
    :param pixel_pos: list indicating [x, y]
    :param depth_map: numpy.ndarray
    :param occluded: int 0 or 1
    :return:
    """
    # TODO to check whether the distance to the tracked object should be represented by the distance or simply depth
    #  now the distance (sqrt(x^2 + y^2 + z^2)) is adopted
    dist_particle = np.linalg.norm(particle[:, :3], axis=1)
    # TODO suppose now the bboxes is a numpy array of shape (n, 6): [[x_0, y_0, z_0, delta_x, delta_y, delta_z], [...], ...]
    sum_dists = []
    for bbox in bboxes:
        bbox_center = np.array([bbox[0]+bbox[3]/2, bbox[1]+bbox[4]/2, bbox[3]+bbox[5]/2])
        dist_particle_bbox_center = np.linalg.norm(particle[:, :3] - bbox_center, axis=1)
        sum_dist = np.sum(dist_particle_bbox_center)
        sum_dists.append(sum_dist)
    max_id = sum_dists.index(max(sum_dists))
    # TODO suppose only pixels found in the bounding box make sense, right ???
    if pixel_pos[0] > bboxes[max_id][0] & pixel_pos[0] < bboxes[max_id][0] + bboxes[max_id][3] & \
            pixel_pos[1] > bboxes[max_id][1] & pixel_pos[1] < bboxes[max_id][1] + bboxes[max_id][4]:
        # p(a^i|r)
        prob_ai_given_r = stats.norm(dist_particle, std_m).pdf(depth_map[pixel_pos[1], pixel_pos[0]])
        # p(b^i|a^i, o^i)
        if occluded == 0:
            pass
        else:
            pass

    else:
        prob_ai_given_r = 0



