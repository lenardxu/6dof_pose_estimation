from pf.utils.config import cfg

import numpy as np
from numpy.random import uniform
from scipy import stats
from math import exp


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

def get_depth_from_pixel(pixel_pos, bbox):
    """
    compute the depth of given pixel
    :param pixel_pos: list indicating the position of the given pixel
    :param bbox: tuple containing the info of bounding box including the depth info for each pixel within it
    :return: float the depth of given pixel
    """
    delta_x = pixel_pos[0] - bbox[0][0]
    delta_y = pixel_pos[1] - bbox[0][1]
    print(f"delta x & delta y is {delta_x} & {delta_y}")
    return bbox[1][delta_x, delta_y]

def likelihood_with_occlusion(particle, bboxes, pixel_pos, depth_map, occluded):
    """
    compute the likelihood for each pixel given pose with considering the potential occlusion
    :param particle: numpy.ndarray particles of shape (N, 7)
    # TODO to check how the predicted depth array looks, i.e., array of shape (640, 480) with most equal zero or (delta_x, delta_y)???
    :param bboxes: list containing bbox info of detected object: (((x_0, y_0, delta_x, delta_y),np.array((delta_x, delta_y))), ...)
    :param pixel_pos: list indicating (x, y)
    :param depth_map: numpy.ndarray indicating the depth info of each pixel of the image
    :param occluded: int 0 or 1
    :return: np.array of shape (N,)
    """
    prob_likelihood_with_occlusion = 1
    # TODO to check whether the distance to the tracked object should be represented by the distance or simply depth
    #  now the distance (sqrt(x^2 + y^2 + z^2)) is adopted
    dist_particle = np.linalg.norm(particle[:, :3], axis=1)
    # TODO suppose now the bboxes is a list containing bbox info of detected object: [((x_0, y_0, delta_x, delta_y),np.array(Z)), ...]
    sum_dists = []  # initialize the list containing all the additive distances between bbox center and all particles
    bboxes_has_pixel = []  # initialize the list containing the bbox including the given pixel
    mean_bboxes_depth_ids = {}  # initialize the dict corresponding to the given pixel with key: id from bboxes list
                                # and value: mean depth value of all pixels inside the bbox
    for i, bbox in enumerate(bboxes):
        mean_bbox_depth = bbox[1][round(bbox[0][2]/2.), round(bbox[0][3]/2.)]
        bbox_center = np.array([bbox[0][0]+bbox[0][2]/2., bbox[0][1]+bbox[0][3]/2., mean_bbox_depth])
        dist_particle_bbox_center = np.linalg.norm(particle[:, :3] - bbox_center, axis=1)
        sum_dist = np.sum(dist_particle_bbox_center)
        sum_dists.append(sum_dist)
        if pixel_pos[0] > bbox[0][0] & pixel_pos[0] < bbox[0][0] + bbox[0][2] & \
            pixel_pos[1] > bbox[0][1] & pixel_pos[1] < bbox[0][1] + bbox[0][3]:
            bboxes_has_pixel.append(bbox)
            mean_bboxes_depth_ids[i] = mean_bbox_depth
    min_id = sum_dists.index(min(sum_dists))  # id indicating the bbox with the min distance from all particles
    if bool(mean_bboxes_depth_ids):
        print(f"mean_bboxes_depth_ids is {mean_bboxes_depth_ids}")
        min_depth_id = min(mean_bboxes_depth_ids,
                           key=mean_bboxes_depth_ids.get)  # id indicating the bbox with the min distance from camera
        # TODO suppose only pixels found in the bounding box make sense, right ???
        if pixel_pos[0] > bboxes[min_id][0][0] & pixel_pos[0] < bboxes[min_id][0][0] + bboxes[min_id][0][2] & \
                pixel_pos[1] > bboxes[min_id][0][1] & pixel_pos[1] < bboxes[min_id][0][1] + bboxes[min_id][0][3]:
            pixel_r_bounded_bbox_depth = get_depth_from_pixel(pixel_pos, bboxes[min_id])
            pixel_nearest_bbox_depth = get_depth_from_pixel(pixel_pos, bboxes[min_depth_id])
            pixel_depth = depth_map(pixel_pos[1], pixel_pos[0])
            # p(a^i|r)
            prob_ai_given_r = stats.norm(dist_particle, cfg.std_m).pdf(pixel_r_bounded_bbox_depth)
            # p(b^i|a^i, o^i)
            if occluded == 0:
                if (abs(pixel_r_bounded_bbox_depth - pixel_nearest_bbox_depth)) <= cfg.dist_thres:
                    prob_bi_given_ai_r = 1
                else:
                    prob_bi_given_ai_r = 0
            else:
                if pixel_nearest_bbox_depth < 0 or pixel_nearest_bbox_depth > pixel_r_bounded_bbox_depth + cfg.dist_thres:
                    prob_bi_given_ai_r = 0
                else:
                    prob_bi_given_ai_r = cfg.lambda_ * exp(-cfg.lambda_ * pixel_nearest_bbox_depth) / \
                                         (1 - exp(-cfg.lambda_ * pixel_r_bounded_bbox_depth))
            # p(z^i|b^i)
            if pixel_depth > 0 & pixel_depth < cfg.m:
                prob_zi_given_bi = (1 - cfg.beta) * stats.norm(pixel_nearest_bbox_depth, cfg.std_c).pdf(pixel_depth) + \
                                                                            cfg.beta / cfg.m
            else:
                prob_zi_given_bi = (1 - cfg.beta) * stats.norm(pixel_nearest_bbox_depth, cfg.std_c).pdf(pixel_depth)
            prob_likelihood_with_occlusion *= prob_ai_given_r * prob_bi_given_ai_r * prob_zi_given_bi
        else:
            prob_ai_given_r = cfg.smooth_val
            prob_likelihood_with_occlusion *= prob_ai_given_r
    else:
        prob_ai_given_r = cfg.smooth_val
        prob_likelihood_with_occlusion *= prob_ai_given_r
    return prob_likelihood_with_occlusion




