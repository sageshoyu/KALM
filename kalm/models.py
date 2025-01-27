from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from kalm.net_utils import DiffusionHeadOutDict, KpEncoder, TemporalUnet_diffuser
from kalm.rotation_utils import compute_rotation_matrix_from_ortho6d
from kalm.utils import get_pointcloud, transform_pointcloud

DIFFUSION_PREDICTION_TYPE_POS = "sample"  # sample | epsilon
DIFFUSION_PREDICTION_TYPE_ROT = "sample"  # sample | epsilon
CLIP_SAMPLE = False


@dataclass
class KpEncodedInput:
    keypoints_feats: torch.Tensor  # (B, F)
    keypoints_pos: torch.Tensor  # (B, N, 3)

    @cached_property
    def bs(self):
        return self.keypoints_feats.shape[0]

    @cached_property
    def device(self):
        return self.keypoints_feats.device


class KalmDiffuser(nn.Module):
    def __init__(
        self, unet_embedding_dim=60,
        diffusion_timesteps=100, traj_len=50, n_kp=8, kp_feat_dim=714,
    ):
        super().__init__()

        self.prediction_head = TemporalUnet_diffuser(
            traj_len=traj_len,
            transition_dim=unet_embedding_dim,  # 9 dim action vec
            dim=unet_embedding_dim,
            n_kp=n_kp,
        )
        self.position_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="scaled_linear",
            prediction_type=DIFFUSION_PREDICTION_TYPE_POS,
            clip_sample=CLIP_SAMPLE,
        )
        self.rotation_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=DIFFUSION_PREDICTION_TYPE_ROT,
            clip_sample=CLIP_SAMPLE,
        )
        self.diffusion_steps = diffusion_timesteps
        self.traj_len = traj_len
        self.keypoint_encoder = KpEncoder(in_dim=kp_feat_dim, emb_dim=unet_embedding_dim, n_kp=n_kp)

    def sample_noisy_traj(self, gt_trajectory, timesteps, noise=None):
        # Sample noise
        if noise is None:
            noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)

        # Add noise to the clean trajectories
        pos = self.position_noise_scheduler.add_noise(gt_trajectory[..., :3], noise[..., :3], timesteps)
        rot = self.rotation_noise_scheduler.add_noise(gt_trajectory[..., 3:9], noise[..., 3:9], timesteps)
        noisy_trajectory = torch.cat((pos, rot), -1)
        return noise, noisy_trajectory

    def conditional_sample(self, condition_data, condition_mask, fixed_inputs_oriframe: KpEncodedInput) -> Tuple[torch.tensor, DiffusionHeadOutDict]:
        self.position_noise_scheduler.set_timesteps(self.diffusion_steps)
        self.rotation_noise_scheduler.set_timesteps(self.diffusion_steps)

        # Random noisy_trajectory, conditioned on start-end
        noise = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
        )
        # Noisy condition data
        noise_t = torch.ones((len(condition_data),), device=condition_data.device).long().mul(self.position_noise_scheduler.timesteps[0])
        noise_pos = self.position_noise_scheduler.add_noise(condition_data[..., :3], noise[..., :3], noise_t)
        noise_rot = self.rotation_noise_scheduler.add_noise(condition_data[..., 3:9], noise[..., 3:9], noise_t)
        noisy_condition_data = torch.cat((noise_pos, noise_rot), -1)
        noisy_trajectory = torch.where(condition_mask, noisy_condition_data, noise)

        # Iterative denoising
        timesteps_all = self.position_noise_scheduler.timesteps
        timesteps_all = torch.cat((timesteps_all, torch.tensor(np.array([3] * 5 + [2] * 5 + [1] * 15 + [0] * 15))))
        for t in timesteps_all:
            timesteps = t * torch.ones(len(noisy_trajectory)).to(noisy_trajectory.device).long()
            out_dict = self.prediction_head(noisy_trajectory, timesteps, fixed_inputs_oriframe)
            out_final = out_dict.get_10d_vector()
            pos = self.position_noise_scheduler.step(out_final[..., :3], t, noisy_trajectory[..., :3]).prev_sample
            rot = self.rotation_noise_scheduler.step(out_final[..., 3:9], t, noisy_trajectory[..., 3:9]).prev_sample
            noisy_trajectory = torch.cat((pos, rot), -1)
        noisy_trajectory = torch.cat((noisy_trajectory, out_dict.openess), -1)
        return noisy_trajectory

    def compute_trajectory(self, encoded_inputs: KpEncodedInput) -> Tuple[torch.tensor, DiffusionHeadOutDict]:
        """ """
        # Condition (optional)
        cond_data = torch.zeros(
            (encoded_inputs.keypoints_feats.shape[0], self.traj_len, 9),
            device=encoded_inputs.keypoints_feats.device,
        )
        cond_mask = torch.zeros_like(cond_data).bool()
        # Sample
        trajectory = self.conditional_sample(cond_data, cond_mask, encoded_inputs)
        # trajectory B T 10
        trajectory[..., -1] = trajectory[..., -1].sigmoid()
        return trajectory

    def forward(self, keypoints_feat, keypoints_xyz, gt_trajectory, run_inference=False):
        """
        Arguments:
            keypoints_feat: (B, n_kp, feat_dim)
            keypoints_xyz: (B, n_kp, 3)
            gt_trajectory: (B, T, 10)
        """
        # Prepare inputs
        keypoints_mixed_feat = self.keypoint_encoder(keypoints_feat, keypoints_xyz)  # B D
        keypoints_encoded = KpEncodedInput(keypoints_mixed_feat, keypoints_xyz)

        if run_inference:
            return_traj = self.compute_trajectory(keypoints_encoded)  # tensor (B, T, 10)
            return return_traj

        # Sample a random timestep
        timesteps = torch.randint(
            0, self.position_noise_scheduler.config.num_train_timesteps,
            (gt_trajectory.shape[0],),
            dtype=torch.long,
            device=gt_trajectory.device,
        )
        noise_9d, noisy_trajectory_9d = self.sample_noisy_traj(gt_trajectory[..., :9], timesteps)  # ?. B 1 9
        pred_dict = self.prediction_head(noisy_trajectory_9d, timesteps, keypoints_encoded)
        return noise_9d, pred_dict


class ModelInferenceWrapper:
    def __init__(self, ckpt_path, n_kp=8, traj_len=48, diffusion_timesteps=100, embedding_dim=256):
        # Get model
        model = KalmDiffuser(
            unet_embedding_dim=embedding_dim,
            diffusion_timesteps=diffusion_timesteps,
            traj_len=traj_len,
            n_kp=n_kp,
        )
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(self.device)
        # Load ckpt
        assert os.path.isfile(ckpt_path)
        print(f"=> loading checkpoint '{ckpt_path}'")
        model_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(model_dict["weight"], strict=False)
        print(f"=> loaded successfully '{ckpt_path}' (step {model_dict.get('iter', 0)})")
        del model_dict
        torch.cuda.empty_cache()
        model.eval()
        self.model = model

    @torch.no_grad()
    def predict_traj(self, kp_detection_ret, sample_n_trajectories=1):
        """

        Args:
            kp_detection_ret
        Returns:
            predicted_traj: T 4 4 (model frame)
            predicted_traj_worldframe: T 4 4 (world frame)
        """
        keypoint_feats, keypoint_pcds_selectedframe = (
            kp_detection_ret["kp_feat"],
            kp_detection_ret[f"kp_xyz_3d_pseudoworldframe"],
        )

        camera_extrinsic = kp_detection_ret["camera_pose"]

        keypoint_pcd_center = keypoint_pcds_selectedframe.mean(axis=0)
        keypoint_pcds_selectedframe = keypoint_pcds_selectedframe - keypoint_pcd_center

        predicted_traj_camframe_centered_all = self.model(
            torch.tensor(keypoint_feats).float().unsqueeze(0).repeat(sample_n_trajectories, 1, 1).to(self.device),
            torch.tensor(keypoint_pcds_selectedframe).float().unsqueeze(0).repeat(sample_n_trajectories, 1, 1).to(self.device),
            gt_trajectory=None,
            run_inference=True,
        ).detach().cpu()

        kp_locs_selected_camframe = kp_detection_ret["kp_xyz_3d_camera"]
        kp_locs_selected_worldframe = transform_pointcloud(camera_extrinsic, kp_locs_selected_camframe)

        alignz_rotation_matrix = kp_detection_ret["alignz_rotation_matrix"]

        observed_pointcloud_modelframe = kp_detection_ret["pcd_pseudo_worldframe"]

        all_predicted_traj_modelframe, all_predicted_traj_worldframe = [], []
        for predicted_traj_camframe_centered in predicted_traj_camframe_centered_all:
            # T 8
            predicted_traj_camframe_pos = predicted_traj_camframe_centered[:, :3].numpy() + keypoint_pcd_center
            predicted_traj_camframe_rot_mat3 = compute_rotation_matrix_from_ortho6d(predicted_traj_camframe_centered[:, 3:9]).numpy()  # T 3 3
            predicted_traj_camframe_mat4 = np.repeat(np.eye(4)[np.newaxis, ...], predicted_traj_camframe_pos.shape[0], axis=0)
            predicted_traj_camframe_mat4[:, :3, 3] = predicted_traj_camframe_pos
            predicted_traj_camframe_mat4[:, :3, :3] = predicted_traj_camframe_rot_mat3
            predicted_gripper_isopen = predicted_traj_camframe_centered[:, -1].numpy()
            all_predicted_traj_modelframe.append([predicted_traj_camframe_mat4, predicted_gripper_isopen > 0.5])

            predicted_trajectories_camframe = np.einsum(
                "ij, bjk -> bik",
                np.linalg.inv(alignz_rotation_matrix),
                predicted_traj_camframe_mat4,
            )
            predicted_trajectories_worldframe = np.einsum("ij, bjk -> bik", camera_extrinsic, predicted_trajectories_camframe)
            all_predicted_traj_worldframe.append([predicted_trajectories_worldframe, predicted_gripper_isopen > 0.5])
        return (
            all_predicted_traj_modelframe,
            all_predicted_traj_worldframe,
            observed_pointcloud_modelframe,
            kp_locs_selected_worldframe,
        )
