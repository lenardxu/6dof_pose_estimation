# 6dof_pose_estimation
## Problems with code
1. The current occlusion probabilities update is still not fully correct, since it only iterates by involving two consecutive time steps,
which actually does not satisfy the requirement that the occlusion probabilities update should be performed by involving 
three consecutive time steps.
2. The argument in function `get_depth_from_pixel()` in util_pf.py can be problematic, which e.g. cause the slicing within
this function to be negative -> slicing error.

## Preparing the datasets and toolbox
Ref. readme including following sections: https://github.com/oorrppp2/Particle_filter_approach_6D_pose_estimation
### YCB Video dataset
Download the YCB Video dataset by following the comments [here](https://github.com/yuxng/PoseCNN/issues/81) to your local datasets folder.
### YCB Video toolbox
Download the YCB Video toolbox from [here](https://github.com/yuxng/YCB_Video_toolbox) to 
`<local path to 6D_pose_estimation_particle_filter directory>/CenterFindNet/YCB_Video_toolbox` directory. And unzip 
`results_PoseCNN_RSS2018.zip`.
> $ cd <local path to 6D_pose_estimation_particle_filter directory>/CenterFindNet/YCB_Video_toolbox
> 
> $ unzip unzip results_PoseCNN_RSS2018.zip

## 6d object pose estimation paper & code collection
https://github.com/GeorgeDu/6d-object-pose-estimation