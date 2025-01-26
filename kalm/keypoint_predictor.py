import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from featup.util import norm, pca
from PIL import Image
from torchvision import transforms as T

from kalm.utils import (
    GPT_CLIENT_INFO,
    calc_grid_cell_tlbr,
    draw_grid,
    extract_fpfh_feature,
    farthest_point_sampling,
    get_alignz_campose,
    get_pcd,
    load_pickle_file,
    normalize_to_01scale,
    np_to_o3d,
    preprocess_rgb,
    smooth_mask,
    suppress_stdout_stderr,
    transform_pointcloud,
)
from kalm.vlm_client import GPTClient


@dataclass
class MatchingAlgoParams:
    debug_level: int = 0  # 0 - no debugging. 1 - shows matched points. 2 - shows selected neighborhood
    mradius: float = 0.03
    dino_weight: float = 0.75
    consistent_in_query_ratio: float = 0.7
    thr_goodmatch_ratio: float = 0.6
    goodpoint_ratio: float = 0.5
    n_target_good_keypoint: int = 8
    n_candidate_points: int = 50
    n_neighbor_points: int = 10
    similar_feat_thr: float = 1.005
    similar_points_maxdistance_thr: float = 0.1
    use_gpt_guided_mask_in_query_image: bool = False
    guided_mask_discount_factor: float = 0.9


@dataclass
class ImageRecord(object):
    rgb: np.ndarray
    pcd: np.ndarray
    dino_feature: torch.Tensor
    fpfh_feature: torch.Tensor

    intrinsic: Optional[np.ndarray] = None
    extrinsic: Optional[np.ndarray] = None

    _pcd_valid_mask: int = field(init=False, default=None)

    @property
    def pcd_valid_mask(self):
        if self._pcd_valid_mask is None:
            self._pcd_valid_mask = (self.pcd != 0).any(axis=2).astype(np.float32).astype(bool)
        return self._pcd_valid_mask

    @pcd_valid_mask.setter
    def pcd_valid_mask(self, value):
        self._pcd_valid_mask = value

    part_mask: Optional[np.ndarray] = None
    dino_feature_valid_mask: Optional[np.ndarray] = None
    fpfh_feature_valid_mask: Optional[np.ndarray] = None


class DINOFeatureExtractor:
    def __init__(self, device=None, input_size: int = 224, output_size: int = 256):
        self.transform = T.Compose([T.Resize(input_size), T.ToTensor(), norm])
        self.device = device
        self.output_size = output_size
        with suppress_stdout_stderr():
            self.dino_feature_extractor = torch.hub.load("mhamilton723/FeatUp", "dinov2", use_norm=True).to(self.device)

    def extract_feature_from_image(self, image: np.ndarray) -> torch.tensor:
        """

        Args:
            image:  RGB HWC numpy image 0-255
        Returns:
            feature  DHW. tensor.
        """
        if image.max() <= 1.0:
            print(f"Input should be in range 0-255, RGB order. Please double check the input.")
            image = (image * 255).astype(np.uint8)
        image_pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        extracted_feature = self.dino_feature_extractor(image_tensor)[0]
        extracted_feature = torchvision.transforms.functional.resize(extracted_feature, size=self.output_size)
        return extracted_feature.permute(1, 2, 0).detach()


class KeypointPredictor:
    def __init__(self, config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config is not None:
            self.config = config
            self.feature_extractor = DINOFeatureExtractor(
                device=self.device,
                input_size=config.featup_im_size,
                output_size=config.im_size,
            )
            input_and_distilled_keypoints = load_pickle_file(config.distilled_keypoint_file)

            self.reference_image_record: ImageRecord = input_and_distilled_keypoints["reference_image_record"]
            if isinstance(self.reference_image_record.dino_feature, np.ndarray):
                self.reference_image_record.dino_feature = torch.from_numpy(self.reference_image_record.dino_feature)
            self.reference_image_record.dino_feature = self.reference_image_record.dino_feature.float().to(self.device)
            self.ref_points_xy = input_and_distilled_keypoints["reference_points_xy"]
            self.matching_algo_params = input_and_distilled_keypoints["matching_algo_params"]

            if config.use_gpt_guided_mask_in_query_image:
                self.gpt_client = GPTClient()
                self.gpt_client_info = input_and_distilled_keypoints["gpt_client_info"]
                self.gpt_client_info.gpt_client = self.gpt_client

        self.is_loaded_distilled = config is not None

    @torch.no_grad()
    def find_one_matching_point(
        self,
        image1: ImageRecord,
        image2: ImageRecord,
        image1_reference_point: list,
        matching_algo_params: MatchingAlgoParams,
        guided_mask: np.ndarray = None,
    ):
        """

        Args:
            image1:
            image2:

            image2_invalidmask:
            dino_weight:
            vis:
            guided_mask:    Guided mask provided
            guided_mask_discount_factor: If a pixel is not within the guided mask, the cosine similarity will be discounted


        Returns:
            arg_x On query image
            arg_y On query image
        """
        feat_h, feat_w = image1.dino_feature.shape[:2]

        if guided_mask is not None:
            guided_mask = guided_mask.astype(bool)  # valid pix
        else:
            guided_mask = np.ones((feat_h, feat_w)).astype(bool)

        # get dino cosine similarity
        dino_feat_src = image1.dino_feature[image1_reference_point[1], image1_reference_point[0]].unsqueeze(0)  # 1 384
        dino_feat_query_image = image2.dino_feature.reshape(feat_h * feat_w, image2.dino_feature.shape[2]).contiguous()
        cos_sim_dino_feat = F.cosine_similarity(dino_feat_src, dino_feat_query_image, dim=1)
        cos_sim_dino_feat = cos_sim_dino_feat.reshape(feat_h, feat_w).cpu()
        cos_sim_dino_feat = normalize_to_01scale(cos_sim_dino_feat)
        cos_sim_dino_feat[~guided_mask] *= matching_algo_params.guided_mask_discount_factor
        cos_sim_dino_feat[~image2.pcd_valid_mask] = 0

        # get fpfh cosine similarity
        fpfh_feat_src = image1.fpfh_feature[image1_reference_point[1], image1_reference_point[0]].unsqueeze(0)  # 1 D
        fpfh_feat_query_image = image2.fpfh_feature.reshape(feat_h * feat_w, image2.fpfh_feature.shape[2]).contiguous()
        cos_sim_fpfh_feat = F.cosine_similarity(fpfh_feat_src, fpfh_feat_query_image, dim=1)
        cos_sim_fpfh_feat = cos_sim_fpfh_feat.reshape(feat_h, feat_w)
        cos_sim_fpfh_feat = normalize_to_01scale(cos_sim_fpfh_feat)
        cos_sim_fpfh_feat[~guided_mask] *= matching_algo_params.guided_mask_discount_factor
        cos_sim_fpfh_feat[~image2.pcd_valid_mask] = 0

        # plot the cosine similarity
        argmax = torch.argmax((1 - matching_algo_params.dino_weight) * cos_sim_fpfh_feat + matching_algo_params.dino_weight * cos_sim_dino_feat).item()
        arg_y, arg_x = argmax // feat_h, argmax % feat_h

        if matching_algo_params.debug_level >= 2:
            fig, ax = plt.subplots(1, 4, figsize=(10, 5))
            ax[0].imshow(image1.rgb)
            ax[0].set_title(f"Ref Image")
            ax[1].imshow(cos_sim_dino_feat.numpy())
            ax[1].set_title(f"Cos sim dino")
            ax[2].imshow(cos_sim_fpfh_feat.numpy())
            ax[2].set_title(f"Cos sim fpfh")
            ax[3].imshow(image2.rgb)
            ax[3].set_title(f"Query Image")
            ax[0].scatter([image1_reference_point[0]], [image1_reference_point[1]], s=20, c="r")
            ax[3].scatter([arg_x], [arg_y], s=20, c="r")
            plt.show()
            plt.close()
        return arg_x, arg_y

    def is_feature_distinctive(
        self,
        image1: ImageRecord,
        image2: ImageRecord,
        image1_reference_point: list,
        image2_reference_point: list,
        matching_algo_params: MatchingAlgoParams,
    ):
        ori_dino_feat = image1.dino_feature[image1_reference_point[1], image1_reference_point[0]].unsqueeze(0).unsqueeze(0)  # 1 1 384
        arg_dino_feat = image2.dino_feature[image2_reference_point[1], image2_reference_point[0]].unsqueeze(0).unsqueeze(0)  # 1 1 384
        distance_feat = torch.linalg.norm(ori_dino_feat - arg_dino_feat)

        distance_thr = distance_feat * matching_algo_params.similar_feat_thr
        similar_distance_on_query_mask = (torch.linalg.norm(image2.dino_feature - ori_dino_feat, dim=2) < distance_thr).detach().cpu().numpy()
        points_xyz_similar_feature = image2.pcd[similar_distance_on_query_mask]
        points_xyz_similar_feature = points_xyz_similar_feature[(points_xyz_similar_feature != 0).any(axis=1)]

        # get largeset patch
        points_xyz_similar_feature_o3d = np_to_o3d(points_xyz_similar_feature)
        labels = np.array(points_xyz_similar_feature_o3d.cluster_dbscan(eps=0.01, min_points=20))
        if labels.max() >= 0:
            sorting_list = [(label_id, (labels == label_id).sum()) for label_id in np.unique(labels) if label_id >= 0]
            max_cluster_label = max(sorting_list, key=lambda x: x[1])[0]
            points_xyz_similar_feature = points_xyz_similar_feature[labels == max_cluster_label]
        points_xyz_similar_feature_maxdistance = np.linalg.norm(
            points_xyz_similar_feature - points_xyz_similar_feature[:, np.newaxis],
            axis=2,
        ).max()

        # If the patch is too large, it is not distinctive
        return points_xyz_similar_feature_maxdistance <= matching_algo_params.similar_points_maxdistance_thr

    @torch.no_grad()
    def find_consistent_match(
        self,
        image1: ImageRecord,
        image2: ImageRecord,
        highlight_points_xy: list,
        matching_algo_params: MatchingAlgoParams,
        guided_mask: np.ndarray = None,
    ):
        """
        Args:
            image1:          H W 3
            image2:          H W 3
            dino_feat1:      H W 384
            dino_feat2:      H W 384
            pcd1:           H W 3
            pcd2:           H W 3
            fpfh_feat1:     H W 33
            fpfh_feat2:     H W 33
            ref_x_input1:
            ref_y_input1:
            dino_weight:
            thr_goodmatch_ratio:
            n_sample_points_in_neighborhood:
            mradius:        radius in 3D to define the neighborhood to sample from
            debug_level:

        Returns:
            is_consistent:      Is the ref point a good candidate point. If there is a consistent matching patch, returns True.
            matched_point_yx    The matched point in query image.
        """
        im_h, im_w, _ = image1.rgb.shape
        arg_x1_on_query_im, arg_y1_on_query_im = self.find_one_matching_point(
            image1,
            image2,
            highlight_points_xy,
            matching_algo_params=matching_algo_params,
            guided_mask=guided_mask,
        )

        argmax_image2_xy = [arg_x1_on_query_im, arg_y1_on_query_im]
        good_match_ref_yx = [tuple(highlight_points_xy[::-1])]
        good_match_query_yx = [tuple(argmax_image2_xy[::-1])]

        is_distinct_point = self.is_feature_distinctive(image1, image2, highlight_points_xy, argmax_image2_xy, matching_algo_params)
        if not is_distinct_point:
            return False, good_match_query_yx[0]

        xyz_argmax_seedpoint_in_query_pcd = image2.pcd[arg_y1_on_query_im, arg_x1_on_query_im]
        assert (xyz_argmax_seedpoint_in_query_pcd != 0).any()

        distance_to_query_point_3d = image1.pcd.reshape(-1, 3) - image1.pcd[highlight_points_xy[1], highlight_points_xy[0]]
        sample_from_mask = ((distance_to_query_point_3d**2).sum(axis=1) <= matching_algo_params.mradius**2).reshape(im_h, im_w)
        if image1.part_mask is not None:
            ref_part_mask_resized = smooth_mask(image1.part_mask)
            sample_from_mask = sample_from_mask & ref_part_mask_resized

        if matching_algo_params.debug_level >= 2:
            plt.imshow(sample_from_mask)
            plt.title("neighborhood mask to sample from")
            plt.show()
            plt.close()

        sample_from_yx_pix = np.array(sample_from_mask.nonzero()).T
        if not sample_from_yx_pix.any():
            return False, good_match_query_yx[0]

        n_sample_points_in_neighborhood = min(matching_algo_params.n_neighbor_points, len(sample_from_yx_pix))
        sample_from_point_xyz = np.array([image1.pcd[yy, xx] for yy, xx in sample_from_yx_pix])
        sampled_nearby_points_xyz, sampled_nearby_points_idx = farthest_point_sampling(sample_from_point_xyz, n_sample_points_in_neighborhood, return_idx=True)
        sampled_nearby_points_yx = sample_from_yx_pix[sampled_nearby_points_idx]

        for sampled_nearby_point in sampled_nearby_points_yx:
            y_src, x_src = sampled_nearby_point
            arg_x1_on_query_im, arg_y1_on_query_im = self.find_one_matching_point(
                image1,
                image2,
                [x_src, y_src],
                matching_algo_params=matching_algo_params,
                guided_mask=guided_mask,
            )
            matched_point_xyz = image2.pcd[arg_y1_on_query_im, arg_x1_on_query_im]
            match_query_yx = (arg_y1_on_query_im, arg_x1_on_query_im)
            ref_yx = (y_src, x_src)
            if np.linalg.norm(matched_point_xyz - xyz_argmax_seedpoint_in_query_pcd) <= matching_algo_params.mradius:
                good_match_ref_yx.append(ref_yx)
                good_match_query_yx.append(match_query_yx)

        is_consistent = len(good_match_query_yx) / n_sample_points_in_neighborhood >= matching_algo_params.thr_goodmatch_ratio

        if is_consistent and matching_algo_params.debug_level >= 1:
            [feat1_pca, feat2_pca], _ = pca(
                [
                    image1.dino_feature.permute(2, 0, 1).unsqueeze(0),
                    image2.dino_feature.permute(2, 0, 1).unsqueeze(0),
                ]
            )
            plt.clf()
            plt.close()
            fig, ax = plt.subplots(1, 6, figsize=(15, 5))

            ax[0].imshow(image1.rgb)
            ax[0].set_title("Image1")
            ax[1].imshow(image2.rgb)
            ax[1].set_title("Image2")

            ax[2].imshow(image1.pcd[..., 2])
            ax[2].set_title("Depth 1")
            ax[3].imshow(image2.pcd[..., 2])
            ax[3].set_title("Depth 2")

            ax[4].imshow(feat1_pca[0].permute(1, 2, 0).detach().cpu().numpy())
            ax[4].set_title("DINO Feature Im1")
            ax[5].imshow(feat2_pca[0].permute(1, 2, 0).detach().cpu().numpy())
            ax[5].set_title("DINO Feature Im2")

            cmap = plt.get_cmap("gist_rainbow")
            color = [cmap((i + 1) / (len(good_match_ref_yx) + 1)) for i in range(len(good_match_ref_yx))]
            for i, (y, x) in enumerate(good_match_ref_yx):
                ax[0].scatter(x, y, c=color[i], s=20)
            for i, (arg_y, arg_x) in enumerate(good_match_query_yx):
                ax[1].scatter(arg_x, arg_y, c=color[i], s=20)
            plt.show()
            plt.close()
        return is_consistent, good_match_query_yx[0]

    def obtain_guided_mask(self, image_numpy_query, vis=False):
        self.gpt_client_info.update_query_image(image_numpy_query)
        gpt_guided_grids, prev_messages = self.gpt_client.obtain_GPT_mask_on_query_image_auto(self.gpt_client_info, max_retries=3)
        print(prev_messages[-1]["content"])
        print(f"GPT returned: {gpt_guided_grids}")
        if vis:
            image_with_grid = draw_grid(
                image_numpy_query,
                nr_vertical=self.gpt_client_info.grid_shape[1],
                nr_horizontal=self.gpt_client_info.grid_shape[0],
                resize_to_max_dim=512,
            )
            plt.imshow(image_with_grid)
            plt.axis("off")
            plt.title("Grid on Image")
            plt.show()
            plt.close()

        guided_mask = np.zeros((self.config.im_size, self.config.im_size))
        for grid_id in gpt_guided_grids:
            grid_tlbr = calc_grid_cell_tlbr(
                (self.config.im_siz, self.config.im_siz),
                self.gpt_client_info.grid_shape,
                grid_id - 1,
            )
            top, left, bottom, right = grid_tlbr
            guided_mask[top:bottom, left:right] = 1

        return guided_mask

    @torch.no_grad()
    def find_n_matching_keypoints(
            self,
            image2_record: ImageRecord,
            image2_guide_mask: Optional[np.ndarray] = None,
            discount_neighborhood_pixel_range=5
    ):
        # Find n matching keypoints
        if image2_guide_mask is None:
            image2_guide_mask = np.ones(image2_record.rgb.shape[:2]).astype(bool)
        matched_keypoints_yx = []
        for ref_point_xy in self.ref_points_xy:
            matched_keypoint_yx = self.find_one_matching_point(
                self.reference_image_record,
                image2_record,
                image1_reference_point=ref_point_xy,
                matching_algo_params=self.matching_algo_params,
                guided_mask=image2_guide_mask
            )
            matched_keypoints_yx.append(matched_keypoint_yx)
            # Set the neighorhood of the matched keypoint to be invalid in the guided mask
            y, x = matched_keypoint_yx
            image2_guide_mask[max(0, y - discount_neighborhood_pixel_range):min(image2_record.rgb.shape[0],y + discount_neighborhood_pixel_range),
                                max(0, x - 5):min(image2_record.rgb.shape[1], x + 5)] = False
        return matched_keypoints_yx

    def predict_keypoints_given_training_config(self, query_rgb_im, query_dep_im, intrinsic, extrinsic, query_pcd=None):
        assert self.is_loaded_distilled
        query_rgb = preprocess_rgb(query_rgb_im[np.newaxis, ...], self.config.im_size)[0]  # H W 3
        query_dino_feature = self.feature_extractor.extract_feature_from_image(query_rgb_im)  # H W C
        if query_pcd is None:
            query_pcd = get_pcd(query_dep_im, intrinsic, resize=query_rgb_im.shape[0])  # H W 3
        query_fpfh_feature = extract_fpfh_feature(query_pcd)  # H W 33

        if self.config.use_gpt_guided_mask_in_query_image:
            print("Querying VLM for guidance mask.")
            guided_mask = self.obtain_guided_mask(query_rgb, vis=self.config.vis_level >= 2)
        else:
            guided_mask = np.ones((self.config.im_size, self.config.im_size)).astype(bool)

        query_image_record = ImageRecord(
            rgb=query_rgb,
            pcd=query_pcd,
            dino_feature=query_dino_feature,
            fpfh_feature=query_fpfh_feature,
        )
        print(f"Find matching keypoints.")
        points_pairs_ij = self.find_n_matching_keypoints(query_image_record, image2_guide_mask=guided_mask)

        if self.config.vis_level >= 1:
            fig, ax = plt.subplots(1, 3, figsize=(10, 5))
            ax[0].imshow(self.reference_image_record.rgb)
            ax[0].set_title(f"Ref Image")
            ax[1].imshow(query_pcd[..., 2])
            ax[1].set_title(f"Obs Depth")
            ax[2].imshow(query_image_record.rgb)
            ax[2].set_title(f"Query Image")
            color_map = plt.get_cmap("gist_rainbow")
            colors = [color_map((i + 1) / len(self.ref_points_xy)) for i in range(len(self.ref_points_xy))]
            ax[0].scatter([pt[0] for pt in self.ref_points_xy], [pt[1] for pt in self.ref_points_xy], c=colors, s=20)
            ax[2].scatter([pt[1] for pt in points_pairs_ij], [pt[0] for pt in points_pairs_ij], c=colors, s=20)
            plt.show()
            plt.close()

        updated_campose = get_alignz_campose(extrinsic)
        rotate_matrix = np.linalg.inv(updated_campose).dot(extrinsic)
        rotated_pointcloud = transform_pointcloud(rotate_matrix, query_pcd.reshape(-1, 3)).reshape(*query_pcd.shape[:2], 3)
        keypoint_pcds = [rotated_pointcloud[y, x] for (y, x) in points_pairs_ij]

        query_dino_feats_numpy = query_dino_feature.detach().cpu().numpy()
        keypoint_feats_dino = np.array([query_dino_feats_numpy[:, y, x] for (y, x) in points_pairs_ij])
        keypoint_feats_fpfh = np.array([query_fpfh_feature[:, y, x] for (y, x) in points_pairs_ij])
        keypoint_feats = np.concatenate((keypoint_feats_dino, np.repeat(keypoint_feats_fpfh, 10, axis=1)), axis=1).astype(np.float32)

        return_dict = {
            "query_rgb_resized": query_rgb,
            "query_pcd_resized": query_pcd,
            "depth_im": query_dep_im,
            "intrinsic": intrinsic,
            "camera_pose": extrinsic,
            'kp_yx_2d': np.array(points_pairs_ij),
            "kp_feat": np.array(keypoint_feats),
            'kp_xyz_3d_camera': np.array([query_pcd[y, x] for (y, x) in points_pairs_ij]),
            "kp_xyz_3d_pseudoworldframe": np.array(keypoint_pcds),
            "alignz_rotation_matrix": rotate_matrix,
        }
        return return_dict
