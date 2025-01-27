import random

import numpy as np
import torch
import trimesh
from scipy.interpolate import CubicSpline, interp1d

from torch.utils.data.dataset import Dataset
from kalm.rotation_utils import euler2mat, get_ortho6d_from_rotation_matrix, matrix_to_quaternion, normalise_quat, quaternion_to_matrix


class TrajectoryInterpolator:
    """Interpolate a trajectory to have a fixed length."""

    def __init__(self, use=True, interpolation_length=50):
        self._use = use
        self._interpolation_length = interpolation_length

    def __call__(self, trajectory):
        assert trajectory.shape[1] == 8  # pos, wxyz, gripper
        if not self._use:
            return trajectory

        trajectory = trajectory.numpy()
        # Calculate the current number of steps
        old_num_steps = len(trajectory)

        # Create a 1D array for the old and new steps
        old_steps = np.linspace(0, 1, old_num_steps)
        new_steps = np.linspace(0, 1, self._interpolation_length)

        # Interpolate each dimension separately
        resampled = np.empty((self._interpolation_length, trajectory.shape[1]))
        for i in range(trajectory.shape[1]):
            if i == (trajectory.shape[1] - 1):  # gripper opening
                interpolator = interp1d(old_steps, trajectory[:, i])
            else:
                interpolator = CubicSpline(old_steps, trajectory[:, i])

            resampled[:, i] = interpolator(new_steps)

        resampled = torch.tensor(resampled)
        if trajectory.shape[1] == 8:
            resampled[:, 3:7] = normalise_quat(resampled[:, 3:7])
        return resampled


class KeypointDataset(Dataset):
    def __init__(
        self, dataset_path, num_iters=100000, training=True,
        interpolation_length=50, n_kp=8, vis=False,
        augment_axis=None, return_augment_matrix=False, augment_perturbpoint_numpoint=4, augment_rotate_range=np.pi / 2,
    ):
        self.vis = vis
        self._num_iters = num_iters
        self._training = training
        self._interpolate_traj = TrajectoryInterpolator(interpolation_length=interpolation_length)

        # Collect and trim all episodes in the dataset
        self._episodes = np.load(dataset_path, allow_pickle=True)["data"]
        self._num_episodes = len(self._episodes)
        self.n_kp = n_kp
        self.augment_axis = augment_axis if augment_axis is not None else [0, 1, 2]
        self.return_augment_matrix = return_augment_matrix
        self.augment_perturbpoint_numpoint = augment_perturbpoint_numpoint
        self.rotate_range = augment_rotate_range

        print(f"Created dataset from {dataset_path} with {self._num_episodes}.")
        print(f"Augment axis {self.augment_axis}")

    def __len__(self):
        if self._num_iters is not None:
            return self._num_iters
        return self._num_episodes

    def get_validstep_ids(self, traj_qpos, valid_delta=2e-4):
        """
        Args:
            traj_qpos:
            valid_delta:

        Returns:
            Array of valid entry ids
        """
        truncated_ids = [0]
        for k in range(traj_qpos.shape[0]):
            delta_dist = np.linalg.norm(traj_qpos[k] - traj_qpos[truncated_ids[-1]])
            if delta_dist < valid_delta:
                continue
            truncated_ids.append(k)
        truncated_ids = np.array(truncated_ids)
        return truncated_ids

    def rotate_aug(self, traj):
        """
        Args:
            traj: (T, 4, 4)
        """
        random_tform = torch.eye(4).float()
        roll = np.random.rand() * self.rotate_range - self.rotate_range / 2.0 if 0 in self.augment_axis else 0
        pitch = np.random.rand() * self.rotate_range - self.rotate_range / 2.0 if 1 in self.augment_axis else 0
        yaw = np.random.rand() * self.rotate_range - self.rotate_range / 2.0 if 2 in self.augment_axis else 0
        random_rot = torch.tensor(euler2mat(np.array([roll, pitch, yaw]))).float()  # 3 3
        random_tform[:3, :3] = random_rot
        augmented = torch.bmm(random_tform.unsqueeze(0).repeat(traj.shape[0], 1, 1), traj)
        return random_rot, augmented

    def convert_rot(self, traj):
        """
        Adapted from https://github.com/nickgkan/3d_diffuser_actor
        Args:
            traj: T 8 (3 pos + 4 quat wxyz + 1 gripper openness)

        Returns:
            traj: T 10 (3 pos + 6 ortho + 1 gripper openness)
        """
        traj[..., 3:7] = normalise_quat(traj[..., 3:7])
        pos = traj[..., :3]
        rot = quaternion_to_matrix(traj[..., 3:7])
        grip = traj[..., 7:]
        rot_6d = get_ortho6d_from_rotation_matrix(rot)
        traj = torch.cat([pos, rot_6d, grip], dim=-1)
        return traj

    def __getitem__(self, episode_id):
        episode_id %= self._num_episodes
        episode_data = self._episodes[episode_id]

        # Split RGB and XYZ
        keypoint_feats_dino = episode_data["keypoint_feats_dino"].copy()
        keypoint_feats_fpfh = episode_data["keypoint_feats_fpfh"].copy()
        keypoint_xyz = episode_data["keypoint_xyz"].copy()

        pcd_center = keypoint_xyz.mean(axis=0)
        keypoint_xyz = keypoint_xyz - pcd_center

        ee_poses_trajectory = torch.tensor(episode_data["ee_poses_trajectory"].copy()).float()
        ee_poses_trajectory[:, :3, 3] -= pcd_center

        aug_tform = np.eye(4)

        if self._training:
            # random pertubation first
            if self.augment_perturbpoint_numpoint > 0:
                noise = np.random.randn(self.augment_perturbpoint_numpoint, 3) * 0.04
                perturb_id = random.choices(np.arange(self.n_kp), k=self.augment_perturbpoint_numpoint)
                keypoint_xyz[perturb_id] += noise
                feats_fpfh_perturb = np.random.randn(self.augment_perturbpoint_numpoint, 33) * 0.2
                feats_dino_perturb = np.random.randn(self.augment_perturbpoint_numpoint, 384) * 0.2
                keypoint_feats_dino[perturb_id] += feats_dino_perturb
                keypoint_feats_fpfh[perturb_id] += feats_fpfh_perturb

            aug_tform, ee_poses_trajectory = self.rotate_aug(ee_poses_trajectory)
            keypoint_xyz = torch.bmm(
                aug_tform.unsqueeze(0).repeat(keypoint_xyz.shape[0], 1, 1).float(),
                torch.tensor(keypoint_xyz).unsqueeze(2).float(),
            ).squeeze(2).numpy()

        # Get action tensors for respective frame ids
        action_pos = ee_poses_trajectory[:, :3, 3]
        action_rot_mat = ee_poses_trajectory[:, :3, :3]

        action_rot_quat = matrix_to_quaternion(action_rot_mat)
        action_grip = torch.tensor(episode_data["gripper_openness_trajectory"].copy()).reshape(-1, 1)  # gripper is open
        action_dim8 = torch.cat((action_pos, action_rot_quat, action_grip), dim=1)
        valid_action_ids = self.get_validstep_ids(torch.tensor(episode_data["joint_q_trajectory"]).reshape(action_dim8.shape[0], 7))
        action_dim8 = action_dim8[valid_action_ids]
        action_interpolated_dim8 = self._interpolate_traj(action_dim8)
        action_interpolated_dim10 = self.convert_rot(action_interpolated_dim8)

        if self.vis:
            pcd_obs = trimesh.points.PointCloud(keypoint_xyz)
            pcd_obs.colors = np.array([255, 0, 0])
            pcd_traj = trimesh.points.PointCloud(action_interpolated_dim10[:, :3])
            pcd_traj.colors = np.linspace(
                np.array([0, 50, 0]),
                np.array([250, 250, 9]),
                action_interpolated_dim10.shape[0],
            )
            trimesh.Scene([pcd_obs, pcd_traj]).show()

        ret_dict = {
            "keypoints_feat": torch.cat((torch.tensor(keypoint_feats_dino).float(), torch.tensor(keypoint_feats_fpfh).float().repeat(1, 10)), dim=1),  # K dim
            "keypoints_xyz": torch.tensor(keypoint_xyz).float(),  # K 3
            "gt_trajectory": action_interpolated_dim10.float(),  # (n_frames, 10-dof), target poses
        }
        if self.return_augment_matrix:
            ret_dict.update({"aug_tform": aug_tform})
        return ret_dict


def visualize_trajectory():
    # Visualize the training trajectories
    dset = KeypointDataset("keypoint_files/example/kalmdiffuser_train.npz", augment_axis=[2])
    print(len(dset))
    ret_traj_all = []
    pcd_all = []
    for i in range(200):
        ret_dict = dset.__getitem__(i)
        ret_traj_all.append(ret_dict)
        pcd = trimesh.points.PointCloud(ret_dict["gt_trajectory"][:, :3])
        pcd.colors = np.linspace(
            np.array([0, 15, 0]),
            np.array([0, 250, 250]),
            ret_dict["gt_trajectory"][:, :3].shape[0],
        )
        pcd_all.append(pcd)

    trimesh.Scene(pcd_all).show()


if __name__ == "__main__":
    visualize_trajectory()
