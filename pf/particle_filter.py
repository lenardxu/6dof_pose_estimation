# First, not include the occlusion!

import numpy as np
from numpy.random import randn, random, uniform, multivariate_normal, seed
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.config import cfg
from utils.utils_pf import occlusion_prob, occlusion_update
from utils.utils_general import multinomial_resample, residual_resample, stratified_resample, \
                                systematic_resample, resample_from_index


class ParticleFilter:
    """
    this particle filter applies to a single object
    """
    def __init__(self, N, initial_pose, initial_std):
        """
        construct particle filter class
        :param N: number of particles
        :param initial_pose: list indicating the initial position of the object
        :param initial_std: list indicating the standard deviation for initial position of the object
        """
        self._particles = np.empty((N, 7))  # position: X, Y, Z; orientation (quaternion): x, y, z, w
        self._N = N
        # specify limits
        self._X_lim = cfg.STATE_LIMIT[0]
        self._Y_lim = cfg.STATE_LIMIT[1]
        self._Z_lim = cfg.STATE_LIMIT[2]
        self._u_lim = cfg.STATE_LIMIT[3]  # since the upper limit for u1, u2, u3 is the same
        # set particles according to given limits and assumed initial distribution
        # TODO to get the (strong) prior knowledge from bounding box predicted by object detector over initial pose from
        #  dataset w.r.t. initial_position & std
        self.set_particles(init_position=None, init_std=None)

        # distribute particles randomly with uniform weight
        self._weights = np.empty(self._N)
        self._weights.fill(1. / self._N)

        # init prob of being occluded for each pixel
        # TODO to check whether it's reasonable to set initial prob of being occluded directly
        self._occl_prob = 1 - cfg.init_vis_prob

    @property
    def N(self):
        """Get value of number of particles"""
        return self._N

    @N.setter
    def N(self, value):
        """Set (i.e., alter) number of particles afterwards, raise error if negative or zero"""
        if value > 0:
            self._N = value
        else:
            raise ValueError("Number of particles must be positive")

    def predict(self, action, std):
        # TODO instead of using bbox for prediction of state / pose, should better use the given action signal w.r.t.
        #  translational velocity and rotational velocity (under const velocity assumption). std is the standard
        #  deviation of specified gaussian distribution - vector (sigma_x, sigma_y, sigma_z, <quaternion - dim:4>)
        """
        predict the pose of particles when moving based on bounding boxes of objects and corresponding standard
        deviations in each position
        :param action: ???
        :param std: standard deviations of middle points of bounding boxes
        """
        # TODO suppose action is a list containing v_x, v_y, v_z, w_x, w_y, w_z


    def update(self, bbox, std):
        pass

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
        return mu, var

    def set_particles(self, init_position=None, init_std=None):
        """
        set the initial states (position + orientation in quaternion) represented by particles
        :param init_position: list indicating only the position of object
        :param init_td: list indicating only the standard deviation w.r.t the position of object
        :return: None
        """
        if init_position is not None:
            self._particles[:, 0] = init_position[0] + (randn(self.N) * init_std[0])
            self._particles[:, 1] = init_position[1] + (randn(self.N) * init_std[1])
            self._particles[:, 2] = init_position[2] + (randn(self.N) * init_std[2])
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




