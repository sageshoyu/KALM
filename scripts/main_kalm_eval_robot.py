import argparse
import os
import os.path
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from kalm.configs.model_config import EE2CAM, config_mapping
from kalm.keypoint_predictor import KeypointPredictor
from kalm.models import ModelInferenceWrapper
from kalm.robot_utils import initialize_robot_policy
from kalm.utils import get_pointcloud


def visualize_prediction(
    keypoint_predictor: KeypointPredictor,
    query_rgb,
    query_xyz,
    keypoint_pred_2d,
    keypoint_pred_3d,
    observed_pointcloud,
    all_predicted_trajectories,
):
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(cv2.resize(keypoint_predictor.reference_image_record.rgb, (256, 256)))
    ax[0].set_title(f"Ref Image")
    ax[1].imshow(query_xyz[..., 2])
    ax[1].set_title(f"Obs Depth")
    ax[2].imshow(query_rgb)
    ax[2].set_title(f"Query Image")
    color_map = plt.get_cmap("gist_rainbow")
    colors = [color_map((i + 1) / len(keypoint_predictor.ref_points_xy)) for i in range(len(keypoint_predictor.ref_points_xy))]
    ax[0].scatter(
        [pt[0] for pt in keypoint_predictor.ref_points_xy],
        [pt[1] for pt in keypoint_predictor.ref_points_xy],
        c=colors,
        s=20,
    )
    ax[2].scatter(
        [pt[0] for pt in keypoint_pred_2d],
        [pt[1] for pt in keypoint_pred_2d],
        c=colors,
        s=20,
    )
    plt.show()
    plt.close()

    # visualize 3d
    trimesh_keypoints = trimesh.points.PointCloud(keypoint_pred_3d)
    trimesh_keypoints.colors = np.array([255, 0, 0])
    trimesh_obspcd = trimesh.points.PointCloud(vertices=observed_pointcloud.reshape(-1, 3)[::10], colors=query_rgb.reshape(-1, 3)[::10])
    vis_list = [trimesh_keypoints, trimesh_obspcd]

    color_map = plt.get_cmap("gist_rainbow")
    for pred_traj_i, pred_traj in enumerate(all_predicted_trajectories):
        trimesh_traj = trimesh.points.PointCloud(pred_traj[:, :3, 3])
        colors = (np.array(color_map((pred_traj_i + 1) / len(all_predicted_trajectories)))[:3] * 255).astype(np.uint8)
        trimesh_traj.colors = np.linspace(colors, np.array([0, 255, 0]), num=pred_traj.shape[0])
        vis_list.append(trimesh_traj)
    trimesh.Scene(vis_list).show()


def run_policy(robot_policy, keypoint_predictor, trajectory_predictor, eval_config):
    robot_state_cur = robot_policy.robot_controller.get_current_joint_confs()
    current_ee_pose, current_joint_conf = (
        robot_state_cur["ee_pose"],
        robot_state_cur["qpos"],
    )
    time.sleep(1)  # wait for the robot to stabilize

    extrinsic = current_ee_pose.dot(EE2CAM)

    # Sample trajectories
    all_results = []
    for matching_proposal_i in range(eval_config.n_matching_proposals):
        print(f">>>>>>>>>>>>>>>> Evaluating matching proposal {matching_proposal_i}")
        result_matching_proposal_i = []

        rgb_im, dep_im, intrinsic = robot_policy.capture_image()

        kp_detection_ret = keypoint_predictor.predict_keypoints_given_training_config(rgb_im, dep_im, intrinsic, extrinsic)

        (all_predicted_trajectories_modelframe, all_predicted_trajectories_worldframe, observed_pointcloud_modelframe,
         kp_locs_selected_worldframe) = trajectory_predictor.predict_traj(kp_detection_ret, sample_n_trajectories=eval_config.sample_n_trajectories)

        pointcloud_worldframe = get_pointcloud(dep_im, intrinsic, extrinsic, near=eval_config.depth_near, frame="world")
        obstacle_cloud_in_world_frame_ = pointcloud_worldframe.reshape(-1, 3)
        obstacle_cloud_in_world_frame = obstacle_cloud_in_world_frame_[(obstacle_cloud_in_world_frame_ != 0).all(axis=1)]
        not_close_to_kp = (
            np.linalg.norm(
                obstacle_cloud_in_world_frame - kp_locs_selected_worldframe[:, np.newaxis],
                axis=2,
            )
            >= 0.05
        ).T.all(axis=1)
        obstacle_cloud_in_world_frame_proposal_i = obstacle_cloud_in_world_frame[not_close_to_kp]

        if eval_config.vis_level >= 2:
            visualize_prediction(
                keypoint_predictor,
                kp_detection_ret["query_rgb_resized"],
                kp_detection_ret["query_pcd_resized"],
                kp_detection_ret["kp_xy_2d"],
                kp_detection_ret["kp_xyz_3d_pseudoworldframe"],
                observed_pointcloud_modelframe,
                [x[0] for x in all_predicted_trajectories_modelframe],
            )

        for traj_i, (
            predicted_ee_poses_in_world_frame_mat4,
            gripper_isopen_selected,
        ) in enumerate(all_predicted_trajectories_worldframe):

            ending_conf_prepose, arm_path_to_traj_ee_0, gripper_path_to_traj_ee_0 = robot_policy.compute_arm_path_to_trajee0(
                [predicted_ee_poses_in_world_frame_mat4[0]],
                [gripper_isopen_selected[0]],
                obstacle_cloud_in_world_frame_proposal_i,
                start_conf=current_joint_conf[:7],
                dense_interpolate_fac=5,
            )

            if arm_path_to_traj_ee_0 is None:
                print(f">>Sampled traj {traj_i} motion planning failed.")
                result_matching_proposal_i.append(None)
            else:
                print(f">>Sampled traj {traj_i} motion planning succeeded.")
                result_matching_proposal_i.append(
                    [
                        kp_detection_ret["query_rgb_resized"],
                        kp_detection_ret["query_pcd_resized"],
                        kp_detection_ret["kp_xy_2d"],
                        kp_locs_selected_worldframe,
                        ending_conf_prepose,
                        arm_path_to_traj_ee_0,
                        gripper_path_to_traj_ee_0,
                        predicted_ee_poses_in_world_frame_mat4,
                        gripper_isopen_selected,
                        kp_detection_ret["kp_feat"],
                    ]
                )
        all_results.append(result_matching_proposal_i)
        if 1 - result_matching_proposal_i.count(None) / len(result_matching_proposal_i) > 0.8:
            break

    # Heuristic
    path_exist_ratio = [1 - proposal_i_list.count(None) / len(proposal_i_list) for proposal_i_list in all_results]
    if max(path_exist_ratio) < eval_config.worst_ratio:
        return None
    best_matching_idx = np.argmax(path_exist_ratio)
    print(f"Best in {len(all_results)}: Traj {best_matching_idx}. Path exist ratio {path_exist_ratio[best_matching_idx]:.3f}")
    best_matching_predtrajs = all_results[best_matching_idx]

    for traj_i, pred_traj_i in enumerate(best_matching_predtrajs):
        if pred_traj_i is None:
            continue
        (
            resized_query_image,
            cropped_query_pcd,
            kp_xy_on_image,
            kp_locs_selected_worldframe,
            ending_conf_prepose,
            arm_path_to_traj_ee_0,
            gripper_path_to_traj_ee_0,
            predicted_ee_poses_in_world_frame_mat4,
            gripper_isopen_selected,
            kp_feats_selected,
        ) = pred_traj_i
        assert arm_path_to_traj_ee_0 is not None

        if eval_config.vis_level >= 1:
            visualize_prediction(
                keypoint_predictor,
                resized_query_image,
                cropped_query_pcd,
                kp_xy_on_image,
                kp_locs_selected_worldframe,
                pointcloud_worldframe,
                [predicted_ee_poses_in_world_frame_mat4],
            )

        if eval_config.wait_gripper:
            gripper_repeat_n = 15
            gripper_open_idx = (gripper_isopen_selected < 0.5).tolist().index(True)
            pre_open = predicted_ee_poses_in_world_frame_mat4[:gripper_open_idx]
            after_open = predicted_ee_poses_in_world_frame_mat4[gripper_open_idx:]
            open_static = np.repeat(
                predicted_ee_poses_in_world_frame_mat4[gripper_open_idx][np.newaxis, ...],
                gripper_repeat_n,
                axis=0,
            )
            traj_gripper_static_traj = np.concatenate((pre_open, open_static, after_open), axis=0)

            pre_open_gripper = gripper_isopen_selected[:gripper_open_idx]
            after_open_gripper = gripper_isopen_selected[gripper_open_idx:]
            gripper_isopen_selected = np.concatenate(
                (pre_open_gripper, np.zeros(gripper_repeat_n), after_open_gripper),
                axis=0,
            )
        else:
            traj_gripper_static_traj, gripper_isopen_selected = (
                predicted_ee_poses_in_world_frame_mat4,
                gripper_isopen_selected,
            )

        # Execution
        robot_policy.robot_controller.execute_joint_impedance_path(arm_path_to_traj_ee_0, gripper_isopen=gripper_path_to_traj_ee_0)
        robot_policy.robot_controller.execute_cartesian_impedance_path(traj_gripper_static_traj, gripper_isopen_selected, speed_factor=4)

        if eval_config.goback_with_rrt:
            _, arm_path_to_home, gripper_path_to_home = robot_policy.compute_arm_path_to_trajee0(
                [current_ee_pose],
                [False],
                None,
                start_conf=robot_policy.robot_controller.get_current_joint_confs()["qpos"][:7],
                dense_interpolate_fac=5,
            )
            robot_policy.robot_controller.execute_joint_impedance_path(arm_path_to_home, gripper_isopen=gripper_path_to_home)

        run_statistics = {
            "rgb_im_raw": rgb_im,
            "dep_im_raw": dep_im,
            "rgb_im": resized_query_image,
            "dep_im": cropped_query_pcd,
            "intrinsic": intrinsic,
            "ee_pose": current_ee_pose,
            "init_qpos": current_joint_conf,
            "keypoints_3d": kp_locs_selected_worldframe,
            "keypoints_2d": kp_xy_on_image,
            "traj_mat4": predicted_ee_poses_in_world_frame_mat4,
            "keypoint_feats": kp_feats_selected,
        }
        return run_statistics

    return None


def free_motion(config, robot_policy):
    # Test generalization to different camera views
    while True:
        print("Enter free motion mode")
        robot_policy.robot_controller.free_motion(gripper_open=True, timeout=config.freemotion_timeout)
        rgb_im, dep_im, intrinsic = robot_policy.capture_image()
        assert dep_im.max() < 500  # range in m (not mm)

        im_h, im_w = rgb_im.shape[:2]
        min_hw = min(im_h, im_w)
        center_crop_rgb = rgb_im[
            (im_h - min_hw) // 2 : (im_h - min_hw) // 2 + min_hw,
            (im_w - min_hw) // 2 : (im_w - min_hw) // 2 + min_hw,
        ]
        plt.imshow(center_crop_rgb)
        plt.show(block=False)
        plt.pause(2.0)
        plt.close()
        continue_or_not = input(f'Continue? Press "Enter" to re-enter free moving mode. Press "c" to capture\n')
        if continue_or_not.strip().lower().startswith("c"):
            return


def benchmarking(benchmarking_config):
    robot_policy = initialize_robot_policy(debug=False)
    keypoint_predictors, trajectory_predictors = [], []
    for config in benchmarking_config.individual_configs:
        keypoint_predictors.append(KeypointPredictor(config=config))
        trajectory_predictors.append(ModelInferenceWrapper(config.ckpt_path, n_kp=config.num_keypoints))
    if not os.path.exists(benchmarking_config.save_dir):
        os.makedirs(benchmarking_config.save_dir)

    for run_num in range(benchmarking_config.eval_iter):
        free_motion(benchmarking_config, robot_policy)
        init_qpos = robot_policy.robot_controller.get_current_joint_confs()["qpos"].copy()

        for config_i, (keypoint_predictor, trajectory_predictor, config) in enumerate(
            zip(
                keypoint_predictors,
                trajectory_predictors,
                benchmarking_config.individual_configs,
            )
        ):
            statistics_config_i = run_policy(robot_policy, keypoint_predictor, trajectory_predictor, config)
            if statistics_config_i is None:
                print(f"Config {config_i} execution error.")
                statistics_config_i = {}
            np.savez_compressed(os.path.join(benchmarking_config.save_dir,f"iter_{run_num:02d}_config_{config_i:02d}"),statistics_config_i)
            robot_policy.robot_controller.reset_joint_to(init_qpos, gripper_open=True)
            print(f"Robot going home.")
            input("keep going?")
    print("Going home.")


class EvalConfig(dict):
    def __getattr__(self, val):
        return self[val]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="debug", type=str, help="directory to dump eval stats")
    parser.add_argument("--task", required=True, type=str, choices=["kcup", "drawer"], help="robot task")
    parser.add_argument("--free_sec", type=float, default=3, help="timeout for each free motion iter")
    parser.add_argument("--eval_iter", type=int, default=50, help="number of iterations")
    args = parser.parse_args()

    eval_global_config = EvalConfig(
        individual_configs=[config_mapping[args.task]],
        save_dir="eval_robot_runs/" + "test_" + args.task,
        eval_iter=args.eval_iter,
        freemotion_timeout=args.free_sec,
    )
    benchmarking(eval_global_config)
