# Dataset:
# @article{xiang2017posecnn,
# author    = {Xiang, Yu and Schmidt, Tanner and Narayanan, Venkatraman and Fox, Dieter},
# title     = {PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes},
# journal   = {arXiv preprint arXiv:1711.00199},
# year      = {2017}
# }

from numpy.random import seed

from pf.particle_filter import ParticleFilter, draw_save_states
from os import path


def run_pf():
    particles_num = 100
    #TODO to test how the program performs when the strong prior over initial pose is given
    pf = ParticleFilter(particles_num)
    pf.update_right_after_init()
    draw_save_states(pf.particles, pf.img, path.join(pf.output_path, '/%06d.jpg' % (pf.img_index + 1)))
    while pf.img_index < len(pf.imgs):
        pf.predict()
        pf.update()
        # resample if too few effective particles
        if pf.neff() < particles_num / 2:
            # systematic resampling method is adopted by default
            pf.resample()
        mu, var = pf.estimate()
        draw_save_states(pf.particles, pf.img, path.join(pf.output_path, '/%06d.jpg' % (pf.img_index + 1), mu))


if __name__ == '__main__':
    seed(2)
    run_pf()