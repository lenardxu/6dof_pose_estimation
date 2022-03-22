import numpy as np
import random
import os
import glob
import scipy.io as scio
import cv2 as cv
from PIL import Image
from scipy.spatial.transform import Rotation as R

Height = 480
Width = 640
Num = 192
Num_Img = 120

def extract_pose_from_mat(meta_path):
    meta_data = scio.loadmat(meta_path)
    poses = meta_data['poses']
    # select the first object only for demo
    r_t_mat = poses[:, :, 1]  # (3,4)
    r = R.from_matrix(r_t_mat[:3, :3])  # Rotation objects
    r_quat_vec = r.as_quat()  # (4,)
    t_vec = r_t_mat[:3, 3]  # (3,)
    return np.concatenate((t_vec, r_quat_vec))



def generate_particle(x = 0, y = 0, cov_0 = 1, cov_1 = 1, form = 'uniform'):
    res = np.zeros((Num, 7))
    if form == 'uniform':
        num_row = 12
        num_col = 16
        for i in range(Num):
            x = random.randint(0, Width)
            y = random.randint(0, Height)
            res[i][0] = x
            res[i][1] = y
    else:
        mean = (x, y)
        cov = np.array([[cov_0, 0], [0,cov_1]])
        x = np.random.multivariate_normal(mean, cov, (Num,), 'raise')
        res[:,:2] = x
    return res

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

def writr_to_video(path_out):
    img_array = []
    size = tuple()
    for filename in os.listdir(path_out):
        file_path = os.path.join(path_out, filename)
        img = cv.imread(file_path)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv.VideoWriter(os.path.join(path_out, 'project.avi'), cv.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def main():
    path_out_root = 'D:/PGM/'
    cov_0_lst = np.arange(720, 1, -6)
    cov_1_lst = np.arange(480, 1, -4)
    info_mat = glob.glob(r'D:/学习类/Semester3/PGM/project/Data/0044/*.mat')
    info_img = glob.glob(r'D:/学习类/Semester3/PGM/project/Data/0044/*color.png')
    chosed_mat = info_mat[:Num_Img]
    chosed_img = info_img[:Num_Img]
    final_p = []
    for i in range(Num_Img):
        if i == 0:
            tmp_p = generate_particle()
        else:
            meta_path = chosed_mat[i]
            tr = extract_pose_from_mat(meta_path)
            x = tr.flatten()[0]
            y = tr.flatten()[1]
            cov_0 = cov_0_lst[i]
            cov_1 = cov_1_lst[i]
            tmp_p = generate_particle(x, y, cov_0, cov_1, 'gauss')
        final_p.append(tmp_p)
    for i in range(Num_Img):
        #img = cv.imread(chosed_img[i])
        img = np.array(Image.open(chosed_img[i]))
        path_out = os.path.join(path_out_root, f'{i}.png')
        draw_save_states(final_p[i], img, path_out)
        
    

