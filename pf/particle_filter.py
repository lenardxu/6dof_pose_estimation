# First, not include the occlusion!

from os import path

import numpy as np
from numpy.random import randn, random, uniform, multivariate_normal, seed
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

import cv2 as cv

from pf.utils.config import cfg
from pf.utils.utils_pf import occlusion_prob, occlusion_update, likelihood_with_occlusion
from pf.utils.utils_general import multinomial_resample, residual_resample, stratified_resample, \
                                systematic_resample, resample_from_index, extract_pose_from_mat, \
                                extract_bboxes_from_dataset


def draw_save_states(particles, img, path_out, mu=None):
    """draw the current particle to the input image and save"""
    # def draw_circle(x):
    #     cv.circle(img, (x[0], x[1]), 1, (0, 0, 255), 2)
    # v_draw_circle = np.vectorize(draw_circle)
    # v_draw_circle(particles)
    for particle in particles:
        cv.circle(img, (int(particle[0]), int(particle[1])), 1, (0, 0, 255), 4)
    if mu!=None:
        cv.circle(img, (mu[0], mu[1]), 1, (0, 255, 0), 4)
    cv.imwrite(path_out, img)


class ParticleFilter:
    """
    This particle filter applies to a single object
    """
    def __init__(self, N, img_path=cfg.img_path, output_path=cfg.output_path):
        """
        construct particle filter class
        :param N: number of particles
        :param initial_pose: list indicating the initial position of the object
        :param initial_std: list indicating the standard deviation for initial position of the object
        """

        self.img_index = 0
        self.imgs = [path.join(img_path, '%06d-color.png' % (i + 1)) for i in range(cfg.test_no)]
        self.depth_imgs = [path.join(img_path, '%06d-depth.png' % (i + 1)) for i in range(cfg.test_no)]
        self.meta_files = [path.join(img_path, '%06d-meta.mat' % (i + 1)) for i in range(cfg.test_no)]
        self.box_files = [path.join(img_path, '%06d-box.txt' % (i + 1)) for i in range(cfg.test_no)]
        print(self.imgs[0])
        print('processing image: %06d.png' % (self.img_index + 1))
        img_first = cv.imread(self.imgs[0])
        self.output_path = output_path
        self._particles = np.empty((N, 7))  # position: X, Y, Z; orientation (quaternion): x, y, z, w
        self._N = N
        # specify limits
        self._X_lim = cfg.STATE_LIMIT[0]
        self._Y_lim = cfg.STATE_LIMIT[1]
        self._Z_lim = cfg.STATE_LIMIT[2]
        self._u_lim = cfg.STATE_LIMIT[3]  # since the upper limit for u1, u2, u3 is the same
        # set particles according to given limits and assumed initial distribution
        print('processing image: %06d-meta.mat' % (self.img_index + 1))
        self.init_pos = extract_pose_from_mat(self.meta_files[0])
        self.set_particles(init_position=self.init_pos[:3], init_std=cfg.std_pos_init)
        draw_save_states(self._particles, img_first, path.join(self.output_path, '000001.jpg'))

        # distribute particles randomly with uniform weight
        self._weights = np.empty(self._N)
        self._weights.fill(1. / self._N)

        # init prob of being occluded for each pixel according to two different states of occlusion
        # TODO to check whether it's reasonable to set initial prob of being occluded directly
        self._occl_prob = [np.full((cfg.DIM[1], cfg.DIM[0], self._N), 1-cfg.init_vis_prob),
                           np.full((cfg.DIM[1], cfg.DIM[0], self._N), 1-cfg.init_vis_prob)]
        # TODO currently there is only iteration invovling one time step instead of two time steps
        self._occl_prob_next_t = [np.full((cfg.DIM[1], cfg.DIM[0], self._N), 1-cfg.init_vis_prob),
                                  np.full((cfg.DIM[1], cfg.DIM[0], self._N), 1-cfg.init_vis_prob)]


    @property
    def N(self):
        """get value of number of particles"""
        return self._N


    @N.setter
    def N(self, value):
        """set (i.e., alter) number of particles afterwards, raise error if negative or zero"""
        if value > 0:
            self._N = value
        else:
            raise ValueError("Number of particles must be positive")

    @property
    def particles(self):
        """get value of number of particles"""
        return self._particles


    def predict(self):
        """
        predict the pose of particles when moving based on bounding boxes of objects and corresponding standard
        deviations in each position, not including occlusion propagation for better arrangement
        """
        if self.img_index < len(self.imgs) - 1:
            self.img_index += 1
        self.img = cv.imread(self.imgs[self.img_index])
        self.meta_file = self.meta_files[self.img_index]
        gt_input = extract_pose_from_mat(self.meta_file)
        self._particles[:, 0] = gt_input[0] + (randn(self.N) * cfg.std_pos)
        self._particles[:, 1] = gt_input[1] + (randn(self.N) * cfg.std_pos)
        self._particles[:, 2] = gt_input[2] + (randn(self.N) * cfg.std_pos)
        #TODO better to set some manual noise onto the give gt input
        self._particles[:, 3] = gt_input[3]
        self._particles[:, 4] = gt_input[4]
        self._particles[:, 5] = gt_input[5]
        self._particles[:, 6] = gt_input[6]
        #draw_save_states(self._particles, self.img, self.output_path + '/%06d.jpg' % (self.img_index + 1))


    def update_right_after_init(self):
        #TODO since we're target at single object tracking, and the we can detect the desired one using object detector,
        # why should still use multiple detected bboxes and then use r to find the nearest one?
        """
        Update the weight of each particle according to computed likelihood incl. occlusion right after initialization
        since the update formula is a little different from the following iterations
        :param depth_map: numpy.ndarray indicating the depth info of each pixel of the image
        :return: None
        """
        self.predict()
        # print(self.imgs[self.img_index])
        # print(self.depth_imgs[self.img_index])
        # print(self.box_files[self.img_index])
        # TODO 10000. can be afterwards be replaced with read value from .mat file
        depth_map = np.array(Image.open(self.depth_imgs[self.img_index])) / 10000.
        log_likelihood = np.zeros((self.N,))  # (N,)

        bboxes = extract_bboxes_from_dataset(self.meta_files[self.img_index],
                                             self.depth_imgs[self.img_index],
                                             self.box_files[self.img_index])
        #TODO for testing only
        cnt = 0
        for x in range(cfg.DIM[0]):  # x is horizontal direction, different from convention
            for y in range(cfg.DIM[1]):  # y is vertical direction
                likelihood = np.zeros((self.N,))  # (N,)
                for occl_state_curr in [0, 1]:
                    prob_likelihood_curr = likelihood_with_occlusion(self._particles, bboxes, (x, y),
                                                                     depth_map, occl_state_curr)  # (N,)
                    temp = np.zeros((self.N,))  # (N,)
                    for occl_state_prev in [0, 1]:
                        prob_occl_update_curr = occlusion_prob(occl_state_prev, occl_state_curr)  # scalar
                        prob_likelihood_prev = likelihood_with_occlusion(self._particles, bboxes, (x, y),
                                                                         depth_map, occl_state_prev)  # (N,)
                        occl_prob_denom = np.zeros((self.N,))  # (N,)
                        for occl_state_prev_2 in [0, 1]:
                            prob_likelihood_prev_2 = likelihood_with_occlusion(self._particles, bboxes,
                                                                               (x, y), depth_map,
                                                                               occl_state_prev_2)  # (N,)
                            occl_prob_denom += prob_likelihood_prev_2   # (N,)
                        self._occl_prob[occl_state_prev][y, x] = prob_likelihood_prev / occl_prob_denom  # (N,)
                        temp += prob_occl_update_curr * self._occl_prob[occl_state_prev][y, x]  # (N,)
                    likelihood += prob_likelihood_curr * temp  # (N,)
                log_likelihood += np.log(likelihood)  # (N,)
                cnt += 1
                print(f"run pixel {cnt} times")
        self._weights *= np.exp(log_likelihood)  # (N,)

        self._weights += 1.e-300  # avoid round-off to zero
        self._weights /= np.sum(self._weights)  # normalize


    def update(self):
        # TODO to check if the dimension is all right in the following computation when all computations should be based on (N,)
        """
        Update the weight of each particle according to computed likelihood incl. occlusion
        :return: None
        """
        log_likelihood = np.zeros((self.N,))  # (N,)
        # TODO 10000. can be afterwards be replaced with read value from .mat file
        depth_map = np.array(Image.open(self.depth_imgs[self.img_index])) / 10000.

        # bboxes shape: (((x_0, y_0, delta_x, delta_y),np.array((delta_x, delta_y))), ...)
        bboxes = extract_bboxes_from_dataset(self.imgs[self.img_index],
                                             self.depth_imgs[self.img_index],
                                             self.box_files[self.img_index])

        for x in range(cfg.DIM[0]):
            for y in range(cfg.DIM[1]):
                likelihood = np.zeros((self.N,))  # (N,)
                for occl_state_curr in [0, 1]:
                    prob_likelihood_curr = likelihood_with_occlusion(self._particles, bboxes, (x, y),
                                                                     depth_map, occl_state_curr)  # (N,)
                    occl_prob_numer = np.zeros((self.N,))  # (N,)
                    occl_prob_denom = np.zeros((self.N,))  # (N,)
                    temp = np.zeros((self.N,))  # (N,)
                    for occl_state_prev in [0, 1]:
                        prob_occl_update_curr = occlusion_prob(occl_state_prev, occl_state_curr)  # scalar
                        self._occl_prob[occl_state_prev][y, x] = self.compute_occl_prob(bboxes, x, y, depth_map,
                                                                                        occl_state_prev,
                                                                                        occl_prob_numer,
                                                                                        occl_prob_denom)  # (N,)
                        temp += prob_occl_update_curr * self._occl_prob[occl_state_prev][y, x]  # (N,)
                    likelihood += prob_likelihood_curr * temp  # (N,)
                log_likelihood += np.log(likelihood)  # (N,)
        self._weights *= np.exp(log_likelihood)  # (N,)

        self._weights += 1.e-300  # avoid round-off to zero
        self._weights /= np.sum(self._weights)  # normalize


    def compute_occl_prob(self, bboxes, pixel_x, pixel_y, depth_map, occl_state_prev,
                          occl_prob_numer, occl_prob_denom):
        """
        compute the occlusion probability p(o^i_t | r_1_:_t, z_1_:_t, u_1_:_t) for all particles in each pixel at time step t
        :param bboxes: list containing bbox info of detected object: [((x_0, y_0, delta_x, delta_y),np.array((delta_x, delta_y))), ...]
        :param pixel_x: int pixel pos in x direction
        :param pixel_y: int pixel pos in y direction
        :param depth_map: numpy.ndarray indicating the depth info of each pixel of the image
        :param occl_state_prev: int previous occluding state given there are occlusion states of three time steps involved in each iteration
        :param occl_prob_numer: expr the (partial) numerator of occlusion prob which is additive
        :param occl_prob_denom: expr the denominator of occlusion prob which is additive
        :return: the occlusion probability for all particles in each pixel at time step t with shape (N,)
        """
        prob_likelihood_prev = likelihood_with_occlusion(self._particles, bboxes, (pixel_x, pixel_y),
                                                                     depth_map, occl_state_prev)  # (N,)
        for occl_state_double_prev in [0, 1]:
            occl_prob_numer += occlusion_prob(occl_state_double_prev, occl_state_prev) * \
                               self._occl_prob[occl_state_double_prev][pixel_x, pixel_y]  # (N,)
        for occl_state_prev_2 in [0, 1]:
            prob_likelihood_prev_2 = likelihood_with_occlusion(self._particles, bboxes, (pixel_x, pixel_y),
                                                               depth_map, occl_state_prev_2)  # (N,)
            temp = 0  # scalar
            for occl_state_double_prev_2 in [0, 1]:
                temp += occlusion_prob(occl_state_double_prev_2, occl_state_prev_2) * \
                        self._occl_prob[occl_state_double_prev_2][pixel_x, pixel_y]  # (N,)
            occl_prob_denom += prob_likelihood_prev_2 * temp  # (N,)
            occl_prob_element = prob_likelihood_prev * occl_prob_numer / occl_prob_denom  # (N,)
        return occl_prob_element  # (N,)


    def neff(self):
        """
        define the effective number of particles to determine the necessity of resampling
        :return: np.ndarray effective number of particles
        """
        return 1. / np.sum(np.square(self._weights))


    def resample(self, method="systematic"):
        """
        executing resampling based on the specified method
        :param method: str method for resampling - default: systematic resampling
        :return: None
        """
        indexes = np.empty((self.N,))
        if method == "multinomial":
            indexes = systematic_resample(self._weights)
        elif method == "residual":
            indexes = residual_resample(self._weights)
        elif method == "stratified":
            indexes = stratified_resample(self._weights)
        elif method == "systematic":
            indexes = systematic_resample(self._weights)

        resample_from_index(self._particles, self._weights, indexes)
        assert np.allclose(self._weights, 1 / self.N)


    def estimate(self):
        """
        compute the weighted mean and variance based on particles (row: position + orientation in quaternion)
        :return: weighted mean (7,) and variance (7,)
        """
        pose = self._particles[:, :]
        mu = np.average(pose, weights=self._weights, axis=0)
        var = np.average((pose - mu) ** 2, weights=self._weights, axis=0)
        # TODO how to draw the expected value together with the particles?
        return mu, var


    def set_particles(self, init_position=None, init_std=None):
        """
        set the initial states (position + orientation in quaternion) represented by particles
        :param init_position: list indicating only the position of object
        :param init_td: list indicating only the standard deviation w.r.t the position of object
        :return: None
        """
        if init_position is not None:
            self._particles[:, 0] = init_position[0] + (randn(self.N) * init_std)
            self._particles[:, 1] = init_position[1] + (randn(self.N) * init_std)
            self._particles[:, 2] = init_position[2] + (randn(self.N) * init_std)
        else:
            self._particles[:, 0] = uniform(0, self._X_lim, size=self.N)
            self._particles[:, 1] = uniform(0, self._Y_lim, size=self.N)
            self._particles[:, 2] = uniform(0, self._Z_lim, size=self.N)
        # set the orientation (quaternion) part which is assumed to be uniformly distributed
        u_sample_1 = uniform(0, self._u_lim, size=self.N)
        u_sample_2 = uniform(0, self._u_lim, size=self.N)
        u_sample_3 = uniform(0, self._u_lim, size=self.N)
        self._particles[:, 3] = np.sqrt(1 - u_sample_1) * np.sin(2 * np.pi * u_sample_2)
        self._particles[:, 4] = np.sqrt(1 - u_sample_1) * np.cos(2 * np.pi * u_sample_2)
        self._particles[:, 5] = np.sqrt(u_sample_1) * np.sin(2 * np.pi * u_sample_3)
        self._particles[:, 6] = np.sqrt(u_sample_1) * np.cos(2 * np.pi * u_sample_3)




