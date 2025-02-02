import glob
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tap
import torch
import trimesh

import kalm.configs.local_config as local_config
from kalm.keypoint_predictor import DINOFeatureExtractor, ImageRecord, KeypointPredictor, MatchingAlgoParams
from kalm.utils import GPT_CLIENT_INFO
from kalm.utils import approx_equal, calc_grid_cell_tlbr, draw_grid, draw_seg_on_im, extract_fpfh_feature, farthest_point_sampling, get_alignz_campose, plot_points
from kalm.utils import preprocess_pcd, preprocess_rgb, sample_farthest_points_from_selected_masked_area, save_files, transform_pointcloud
from kalm.vlm_client import GRID_SHAPE, TASKNAME2DESC, GPTClient, MaskPredictor


def read_data_from_file(data_path: str, video_subsample_num_frames: int, input_size: int = 256):
    """
    Load the data to extract keypoints.
    NOTE: rgbs / pcds are assumed to be squared. (h == w)
    Args:
        data_path:                  Path to the data file containing the reference video and query images
        video_subsample_num_frames: Number of frames to use in the reference video
        input_size:                 Size of the input image for downstream processing

    Returns:
        reference_video_rgb:  RGB NHWC numpy image 0-255. Reference video
        reference_video_pcd:  Numpy array of point cloud. Corresponding point cloud of the reference_video_rgb, in camera frame
        verification_rgbs: RGB NHWC numpy image 0-255. Images to verify the keypoint proposals
        verification_pcds: Numpy array of point cloud. Corresponding point cloud of the verification_rgbs, in camera frame
        verification_dataset_robot_trajectories
    """
    data = np.load(data_path, allow_pickle=True)
    reference_video = data["reference_video_rgb"]
    selected_frames_idx = np.linspace(0, len(reference_video) - 1, num=video_subsample_num_frames).astype(np.int64)
    reference_video_rgb = preprocess_rgb(reference_video[selected_frames_idx], input_size=input_size)
    reference_video_pcd = preprocess_pcd(data["reference_video_pcd"][selected_frames_idx], input_size=input_size)
    verification_rgbs = preprocess_rgb(
        np.array([verification_demo["observed_rgb"] for verification_demo in data["verification_dataset"]]),
        input_size=input_size,
    )
    verification_pcds = preprocess_pcd(
        [verification_demo["observed_pcd"] for verification_demo in data["verification_dataset"]],
        input_size=input_size,
    )
    verification_dataset_robot_trajectories = data["verification_dataset"]
    return (
        reference_video_rgb,
        reference_video_pcd,
        verification_rgbs,
        verification_pcds,
        verification_dataset_robot_trajectories,
    )


def find_consistent_match_points_on_all_query_images(
    reference_image: ImageRecord,
    verification_images: list[ImageRecord],
    matching_algo_params: MatchingAlgoParams,
    gpt_client_info: GPT_CLIENT_INFO,
    keypoint_predictor: KeypointPredictor,
    vis=False,
    save_dir="debug",
    iter_n=0,
):
    sampled_edge_points_yx = sample_farthest_points_from_selected_masked_area(
        reference_image.rgb,
        reference_image.part_mask,
        reference_image.pcd,
        num_candidate_points=matching_algo_params.n_candidate_points,
    )  # K_refpoint x 2.
    plot_points(
        reference_image.rgb,
        [coord[::-1] for coord in sampled_edge_points_yx],
        save_name_prefix=os.path.join(save_dir, "ref_points"),
        vis=vis,
    )
    ################################################################################

    matched_point_allquery_yx = np.zeros((len(verification_images), len(sampled_edge_points_yx), 2), dtype=np.int64)  # N_image x K_refpoint x 2
    is_good_candidate_allquery_allpoint = np.zeros((len(verification_images), len(sampled_edge_points_yx)))  # N_image x K_refpoint
    for im_i, verification_image in enumerate(verification_images):
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Processing query image {im_i}.")
        if matching_algo_params.use_gpt_guided_mask_in_query_image:
            gpt_client_info.update_query_image(verification_image.rgb)
            gpt_guided_grids, prev_messages = gpt_client_info.gpt_client.obtain_GPT_mask_on_query_image_auto(gpt_client_info, max_retries=3, query_im_index=im_i, iter_n=iter_n)
            print(prev_messages[-1]["content"])
            print(f"GPT returned: {gpt_guided_grids}")

            if vis:
                image_with_grid = draw_grid(
                    verification_image.rgb,
                    nr_vertical=gpt_client_info.grid_shape[1],
                    nr_horizontal=gpt_client_info.grid_shape[0],
                    resize_to_max_dim=512,
                )
                plt.imshow(image_with_grid)
                plt.axis("off")
                plt.title("Grid on Image")
                plt.show()
                plt.close()

            guided_mask = np.zeros(reference_image.rgb.shape[:2])
            for grid_id in gpt_guided_grids:
                grid_tlbr = calc_grid_cell_tlbr(
                    reference_image.rgb.shape[:2],
                    gpt_client_info.grid_shape,
                    grid_id - 1,
                )
                top, left, bottom, right = grid_tlbr
                guided_mask[top:bottom, left:right] = 1
        else:
            guided_mask = None

        for ref_pt_i, (ref_y, ref_x) in enumerate(sampled_edge_points_yx):
            if (reference_image.pcd[ref_y, ref_x] == 0).all():  # TODO pre-filter
                continue
            is_good_candidate, matched_point_yx = keypoint_predictor.find_consistent_match(
                reference_image,
                verification_image,
                highlight_points_xy=[ref_x, ref_y],
                matching_algo_params=matching_algo_params,
                guided_mask=guided_mask,
            )
            is_good_candidate_allquery_allpoint[im_i, ref_pt_i] = is_good_candidate
            matched_point_allquery_yx[im_i, ref_pt_i] = matched_point_yx
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finish processing query image {im_i}.")
    is_good_candidate_eachpoint_ratio_in_all_queries = is_good_candidate_allquery_allpoint.sum(axis=0) / len(verification_images)
    good_candidate_point_idx = np.where(is_good_candidate_eachpoint_ratio_in_all_queries >= matching_algo_params.consistent_in_query_ratio)[0]
    print(f">>> Found {len(good_candidate_point_idx)} good keypoints out of {sampled_edge_points_yx.shape[0]} candidates.")
    print(f"good candidate ratio: {is_good_candidate_eachpoint_ratio_in_all_queries}")
    if len(good_candidate_point_idx) < matching_algo_params.n_target_good_keypoint or len(good_candidate_point_idx) / sampled_edge_points_yx.shape[0] < matching_algo_params.goodpoint_ratio:
        return None
    return (
        good_candidate_point_idx,
        sampled_edge_points_yx,
        matched_point_allquery_yx,
        is_good_candidate_allquery_allpoint,
    )


def masks_from_sam_nms(
    gpt_client: GPTClient,
    mask_predictor: MaskPredictor,
    grid,
    image: np.ndarray,
    grid_shape: tuple[int],
    pcd=None,
    remove_degenerate=True,
    vis=False,
    mask_color_list=None,
    iter_n=0,
):
    masks, points = [], []
    for gpt_return_coordinate in grid:
        grid_tlbr = calc_grid_cell_tlbr(image.shape[:2], grid_shape, gpt_return_coordinate - 1)  # GPT is 1-indexed
        masks1, scores1, points1 = mask_predictor.get_candidate_masks_in_grid(
            image, *grid_tlbr,
            min_distance=10, pcd=pcd, remove_degenerate=remove_degenerate, vis=False,
        )
        for msk, score, pt in zip(masks1, scores1, points1):
            for mask_already_picked in masks:
                if approx_equal(msk, mask_already_picked):
                    break
            else:
                # Add to mask if iou < thr
                masks.append(msk)
                points.append(pt)
    plot_anno = [(i, points[i - 1][0]) for i in range(1, len(masks) + 1)]
    masked_image_all = draw_seg_on_im(image, masks, alpha=0.6, plot_anno=plot_anno)

    print(f"SAM returned {len(masks)} candidate masks")
    if vis:
        plt.imshow(masked_image_all)
        plt.axis("off")
        plt.title("Candidate Masks")
        plt.show()
        plt.close()

    masked_image_list = list()
    for i in range(len(masks)):
        np.save(os.path.join(gpt_client.cache_dir, f"iter_{iter_n:02d}_sammask_{i:02d}.npy"), masks[i])
        masked_image_list.append(draw_seg_on_im(image, [masks[i]], plot_anno=[(i + 1, points[i][0])], cm=mask_color_list))
    return masks, masked_image_list, points


def obtain_mask_from_gpt_and_sam(
    gpt_client: GPTClient,
    mask_predictor: MaskPredictor,
    task_desc: str,
    demonstration_vid_sequence: np.ndarray,
    grid_shape: tuple[int, int],
    pcd=None,
    max_trial_loop=3,
    vis=False,
    gpt_prev_iter_msg=None,
    iter_n=0,
):
    for iter_i in range(max_trial_loop):
        # TODO wrap into class method
        gpt_return_coordinates_firstframe, prev_messages = gpt_client.obtain_GPT_coordinate_auto(task_desc, demonstration_vid_sequence, grid_shape=grid_shape)

        print(prev_messages[-1]["content"])
        print(f"GPT returned first frame: {gpt_return_coordinates_firstframe}")

        if vis:
            image_with_grid = draw_grid(
                demonstration_vid_sequence[0],
                nr_vertical=grid_shape[1],
                nr_horizontal=grid_shape[0],
                resize_to_max_dim=512,
            )
            plt.imshow(image_with_grid)
            plt.axis("off")
            plt.title("Grid on Image")
            plt.show()
            plt.close()

        masks_firstframe, masks_overlayed_firstframe, points_firstframe = masks_from_sam_nms(
            gpt_client,
            mask_predictor,
            gpt_return_coordinates_firstframe,
            demonstration_vid_sequence[0],
            grid_shape,
            pcd=pcd,
            vis=vis,
            mask_color_list=[[0, 0, 1]],
            iter_n=iter_n,
        )
        gpt_return_maskid_firstframe, prev_messages = gpt_client.obtain_GPT_mask_auto(
            prev_messages,
            masks_overlayed_firstframe,
            task_desc=task_desc,
            max_retries=3,
            gpt_prev_iter_msg=gpt_prev_iter_msg,
            iter_n=iter_n,
        )

        if gpt_return_maskid_firstframe == -1:
            print(f"Mask selection loop {iter_i} returns no good mask.")
            continue

        print(prev_messages[-1]["content"])
        print(f"GPT returned: {gpt_return_maskid_firstframe}")
        mask_selected_firstframe = masks_overlayed_firstframe[gpt_return_maskid_firstframe - 1]

        if vis:
            plt.imshow(mask_selected_firstframe)
            plt.show()
            plt.close()
        return masks_firstframe[gpt_return_maskid_firstframe - 1], prev_messages
    return None, None


def main(args):
    task_desc = TASKNAME2DESC[args.task_name]

    (
        reference_video_rgbs,
        reference_video_pcds,
        verification_rgbs,
        verification_pcds,
        verification_dataset_robot_trajectories,
    ) = read_data_from_file(
        args.data_path,
        video_subsample_num_frames=args.ref_seq_len,
        input_size=args.im_size,
    )
    reference_pcd_frame0 = reference_video_pcds[0]

    matching_algo_params = MatchingAlgoParams(
        debug_level=args.debug_level,
        n_candidate_points=args.n_candidate_points,
        n_neighbor_points=args.n_neighbor_points,
        thr_goodmatch_ratio=args.thr_goodmatch_ratio,
        consistent_in_query_ratio=args.consistent_in_query_ratio,
        n_target_good_keypoint=args.n_target_good_keypoint,
        goodpoint_ratio=args.goodpoint_ratio,
        mradius=args.mradius,
        dino_weight=args.dino_weight,
        use_gpt_guided_mask_in_query_image=args.use_gpt_guided_mask_in_query_image,
        guided_mask_discount_factor=args.guided_mask_discount_factor,
        similar_feat_thr=args.similar_feat_thr,
        similar_points_maxdistance_thr=args.similar_points_maxdistance_thr,
        # TODO subgroup
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if hasattr(local_config, "SAM_CHECKPOINT_PATH"):
        checkpoint_path = local_config.SAM_CHECKPOINT_PATH
    else:
        checkpoint_path = None

    gpt_client = GPTClient(cache_dir=args.save_path)
    mask_predictor = MaskPredictor(checkpoint_path, device=device)
    feature_extractor = DINOFeatureExtractor(device=device, input_size=args.featup_im_size, output_size=args.im_size)
    keypoint_predictor = KeypointPredictor()

    print("Computing Image features using FeatUp")  # GPU RAM limit
    reference_image_dino = feature_extractor.extract_feature_from_image(reference_video_rgbs[0])
    verification_rgbs_dino = [feature_extractor.extract_feature_from_image(verification_rgb) for verification_rgb in verification_rgbs]

    print("Computing FPFH features")
    reference_pcd_fpfh = extract_fpfh_feature(reference_pcd_frame0)
    verification_pcds_fpfh = [extract_fpfh_feature(verification_pcd) for verification_pcd in verification_pcds]

    ref_invalid_map = (reference_pcd_frame0 == 0).all(axis=2).astype(np.float32).astype(bool)

    reference_image_record = ImageRecord(
        reference_video_rgbs[0],
        reference_pcd_frame0,
        reference_image_dino,
        reference_pcd_fpfh,
    )
    reference_image_record.pcd_valid_mask = ~ref_invalid_map
    verification_image_records = [
        ImageRecord(
            verification_rgbs[i],
            verification_pcds[i],
            verification_rgbs_dino[i],
            verification_pcds_fpfh[i],
        )
        for i in range(len(verification_rgbs))
    ]

    gpt_prev_iter_mask = None
    for iter_i in range(args.max_iter_num):
        print(f">>>>>>>>>>>>>Finding consistent points. Iteration {iter_i + 1}")
        mask_selected, gpt_returned_msg = obtain_mask_from_gpt_and_sam(
            gpt_client,
            mask_predictor,
            task_desc,
            reference_video_rgbs,
            pcd=reference_pcd_frame0,
            grid_shape=GRID_SHAPE,
            vis=args.debug_level >= 1,
            gpt_prev_iter_msg=gpt_prev_iter_mask,
            iter_n=iter_i,
        )
        gpt_prev_iter_mask = draw_seg_on_im(reference_video_rgbs[0], [mask_selected], cm=[[1, 0, 0]])
        if mask_selected is None:
            continue
        mask_selected[ref_invalid_map] = 0
        ref_image_with_selected_mask = draw_seg_on_im(reference_video_rgbs[0], [mask_selected])
        gpt_client_info = GPT_CLIENT_INFO(
            gpt_client=gpt_client,
            previous_messages=gpt_returned_msg,
            ref_image_with_masks=ref_image_with_selected_mask,
            sequence=reference_video_rgbs,
            query_image=None,
            task_desc=task_desc,
            grid_shape=GRID_SHAPE,
        )
        with open("gpt_client_info.pkl", "wb") as f:
            pickle.dump({
                "mask": mask_selected,
                "info": GPT_CLIENT_INFO(
                    gpt_client=None,
                    previous_messages=gpt_returned_msg,
                    ref_image_with_masks=ref_image_with_selected_mask,
                    sequence=reference_video_rgbs,
                    query_image=None,
                    task_desc=task_desc,
                    grid_shape=GRID_SHAPE,
                 )
            }, f)

        reference_image_record.part_mask = mask_selected
        consistent_matches_on_all_query_ims = find_consistent_match_points_on_all_query_images(
            reference_image_record,
            verification_image_records,
            matching_algo_params,
            gpt_client_info,
            keypoint_predictor,
            vis=args.debug_level >= 1,
            save_dir=args.save_path,
            iter_n=iter_i,
        )
        if consistent_matches_on_all_query_ims is None:
            continue
        (
            good_candidate_point_idx,
            reference_points_yx,
            matched_point_allquery_yx,
            im_i_kp_j_is_good_match,
        ) = consistent_matches_on_all_query_ims

        # Visualize the distilled keypoints
        for query_im_i in range(verification_rgbs.shape[0]):
            print(f"showing match for image {query_im_i}. found {len(good_candidate_point_idx)} matches in iter {iter_i}")
            plot_points_save_prefix = os.path.join(args.save_path, f"query_im_{query_im_i:02d}")
            plot_points(
                reference_video_rgbs[0],
                reference_points_yx[good_candidate_point_idx, ::-1],
                verification_rgbs[query_im_i],
                matched_point_allquery_yx[query_im_i, good_candidate_point_idx, ::-1],
                save_name_prefix=plot_points_save_prefix,
            )
        break
    else:
        print(f"Failed to find good keypoints after {args.max_iter_num} iterations. Exiting...")
        return

    if args.debug_level > 0:
        return

    # Save the distilled keypoints
    save_dir = args.save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    reference_distilled_keypoints_feature = np.array([reference_image_record.dino_feature[y, x].cpu().numpy() for (y, x) in reference_points_yx[good_candidate_point_idx]])
    _, selected_for_training_keypoints_idx = farthest_point_sampling(
        reference_distilled_keypoints_feature,
        num_samples=args.n_kp_in_training_set,
        return_idx=True,
    )
    # Visualize the selected fps keypoints
    for query_im_i in range(verification_rgbs.shape[0]):
        print(f"showing fps result for image {query_im_i}.")
        plot_points(
            reference_video_rgbs[0],
            reference_points_yx[good_candidate_point_idx[selected_for_training_keypoints_idx], ::-1],
            verification_rgbs[query_im_i],
            matched_point_allquery_yx[
                query_im_i,
                good_candidate_point_idx[selected_for_training_keypoints_idx],
                ::-1,
            ],
        )
    with open(os.path.join(save_dir, "reference_keypoints.pkl"), "wb") as f:
        gpt_client_info.gpt_client = None
        reference_image_record.dino_feature = reference_image_record.dino_feature.cpu().numpy()
        saved_dict = {
            "reference_image_record": reference_image_record,
            "reference_points_xy": reference_points_yx[good_candidate_point_idx][selected_for_training_keypoints_idx, ::-1],
            "matching_algo_params": matching_algo_params,
            "gpt_client_info": gpt_client_info,
        }
        pickle.dump(saved_dict, f)

    save_as_kalmdiffuser_record(
        verification_image_records,
        matched_point_allquery_yx[:, good_candidate_point_idx[selected_for_training_keypoints_idx]],
        verification_dataset_robot_trajectories,
        save_path=os.path.join(save_dir, "kalmdiffuser"),
    )
    print("Finished.")


def save_as_kalmdiffuser_record(
    verification_image_records: list[ImageRecord],
    matched_keypoints_vrecord_yx,
    verification_dataset_robot_trajectories,
    save_path,
):
    dataset = []
    for verification_image_record, matched_keypoint_yx, trajectory_data in zip(
        verification_image_records,
        matched_keypoints_vrecord_yx,
        verification_dataset_robot_trajectories,
    ):
        camera_extrinsic = trajectory_data["extrinsic"]

        feats_fpfh_kp = np.array([verification_image_record.fpfh_feature[y, x].numpy() for (y, x) in matched_keypoint_yx])
        feats_dino_kp = np.array([verification_image_record.dino_feature[y, x].cpu().numpy() for (y, x) in matched_keypoint_yx])
        updated_campose = get_alignz_campose(camera_extrinsic)
        rotate_matrix = np.linalg.inv(updated_campose).dot(camera_extrinsic)
        ee_poses_trajectory = np.einsum(
            "ij, bjk -> bik",
            np.linalg.inv(updated_campose),
            trajectory_data["ee_poses_worldframe_trajectory"],
        )

        pointcloud_camera_frame = verification_image_record.pcd
        pointcloud_alignz_frame = transform_pointcloud(rotate_matrix, pointcloud_camera_frame.reshape(-1, 3), set_invalid=True).reshape(*verification_image_record.pcd.shape[:2], 3)
        keypoint_3dloc_alignz_frame = np.array([pointcloud_alignz_frame[y, x] for (y, x) in matched_keypoint_yx])

        if True:
            pcd1 = trimesh.points.PointCloud(pointcloud_alignz_frame.reshape(-1, 3)[::100])
            pcd1.colors = np.array([1.0, 0, 0])
            pcd2 = trimesh.points.PointCloud(camera_extrinsic[:3, 3].reshape(1, 3))
            pcd2.colors = np.array([0, 1.0, 0])
            pcd3 = trimesh.points.PointCloud(ee_poses_trajectory[:, :3, 3])
            pcd3.colors = np.array([0, 1.0, 1.0])
            pcd4 = trimesh.points.PointCloud(keypoint_3dloc_alignz_frame)
            pcd4.colors = np.array([0, 0.5, 1.0])
            trimesh.Scene([pcd1, pcd2, pcd3, pcd4]).show()

        dataset.append({
            "keypoint_xyz": keypoint_3dloc_alignz_frame,
            "keypoint_feats_dino": feats_dino_kp,
            "keypoint_feats_fpfh": feats_fpfh_kp,
            "ee_poses_trajectory": ee_poses_trajectory,
            "gripper_openness_trajectory": trajectory_data["gripper_openness_trajectory"],
            "joint_q_trajectory": trajectory_data["joint_q_trajectory"],
        })

    print(f"Saving to {save_path}_train.npz")
    np.savez_compressed(f"{save_path}_train", data=dataset)
    # for sanity check
    np.savez_compressed(f"{save_path}_val", data=dataset[-2:])
    print(f"Training set saved. {len(dataset)} samples.")


class KeypointDistillArguments(tap.Tap):
    # Data paths
    data_path: str
    save_path: str
    ref_seq_len: int = 10  # How many frames to be sampled from the reference video (for prompting GPT)

    task_name: str
    debug_level: int = 0
    max_iter_num: int = 4
    use_gpt_guided_mask_in_query_image: bool = False

    # Feature extraction parameters
    fpfh_radius: float = 0.03
    im_size: int = 256  # image size
    featup_im_size: int = 448  # image size used by FeatUp

    # Keypoint matching parameters
    mradius: float = 0.03
    dino_weight: float = 0.75
    consistent_in_query_ratio: float = 0.7
    thr_goodmatch_ratio: float = 0.6
    goodpoint_ratio: float = 0.4
    n_target_good_keypoint: int = 8
    n_candidate_points: int = 50
    n_neighbor_points: int = 10
    similar_feat_thr: float = 1.005
    similar_points_maxdistance_thr: float = 0.1
    guided_mask_discount_factor: float = 0.9

    # Training set parameters
    n_kp_in_training_set: int = 8


if __name__ == "__main__":
    args = KeypointDistillArguments().parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    src_file_paths = glob.glob("**.py", recursive=True) + glob.glob("kalm/**.py", recursive=True)
    save_files(src_file_paths, os.path.join(args.save_path, "code"))
    args.save(os.path.join(args.save_path, "opt.json"))
    main(args)
