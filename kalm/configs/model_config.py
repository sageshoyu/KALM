import os.path as osp
import numpy as np

from kalm.configs.path_config import KALM_GIT_ROOT

# Deoxys EE to gripper_camera
EE2CAM = np.array([
    [0.01019998, -0.99989995, 0.01290367, 0.03649885],
    [0.9999, 0.0103, 0.0057, -0.034889],
    [-0.00580004, 0.01280367, 0.99989995, -0.04260014],
    [0.0, 0.0, 0.0, 1.0],
])


class kalm_drawer_config(object):
    # keypoint predictor
    use_gpt_guided_mask_in_query_image = False

    # trajectory predictor
    num_keypoints = 8
    distilled_keypoint_file = osp.join(
        KALM_GIT_ROOT,
        "keypoint_files/example/reference_keypoints.pkl",
    )
    ckpt_path = osp.join(KALM_GIT_ROOT, "scripts/example/001/best.pth")
    sample_n_trajectories = 5

    # misc
    transform_ee2cam = EE2CAM
    depth_near = 0.1
    vis_level = 2
    interpolate_eetraj_speed_factor = 4
    wait_gripper = True
    n_matching_proposals = 2
    worst_ratio = 0.3
    goback_with_rrt = False

    featup_im_size = 448
    im_size = 256
    ignore_collision_distance_rel_kp = 0.05


config_mapping = {
    "drawer": kalm_drawer_config,
}
