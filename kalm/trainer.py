"""Diffuser Actor"""

import numpy as np
import trimesh
from torch.nn import functional as F

import os
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from kalm.dataset import KeypointDataset
from kalm.models import (
    DIFFUSION_PREDICTION_TYPE_POS,
    DIFFUSION_PREDICTION_TYPE_ROT,
    KalmDiffuser,
)


class BaseTrainTester:
    """Basic train/test class to be inherited."""

    def __init__(self, args):
        """Initialize."""
        self.args = args
        if not args.eval_only:
            self.writer = SummaryWriter(log_dir=args.log_dir)

    @staticmethod
    def get_datasets():
        """Initialize datasets."""
        train_dataset = None
        test_dataset = None
        return train_dataset, test_dataset

    def get_loaders(self, collate_fn=default_collate):
        """Initialize data loaders."""

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        # Datasets
        train_dataset, test_dataset = self.get_datasets()
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size, shuffle=True, drop_last=True,
            num_workers=self.args.num_workers, worker_init_fn=seed_worker,
            collate_fn=collate_fn, pin_memory=True, generator=g,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size_val, shuffle=False, drop_last=False,
            num_workers=self.args.num_workers, worker_init_fn=seed_worker,
            collate_fn=collate_fn, pin_memory=True, generator=g,
        )
        return train_loader, test_loader

    @staticmethod
    def get_model():
        """Initialize the model."""
        return None

    @staticmethod
    def get_criterion():
        """Get loss criterion for training."""
        # criterion is a class, must have compute_loss and compute_metrics
        return None

    def get_optimizer(self, model):
        """Initialize optimizer."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.args.lr},
            {"params": [], "weight_decay": 5e-4, "lr": self.args.lr},
        ]
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        for name, param in model.named_parameters():
            if any(nd in name for nd in no_decay):
                optimizer_grouped_parameters[0]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        optimizer = optim.AdamW(optimizer_grouped_parameters)
        return optimizer

    def main(self, collate_fn=default_collate):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, test_loader = self.get_loaders(collate_fn)

        # Check for a checkpoint
        start_iter, best_loss = 0, None
        if self.args.checkpoint:
            print(self.args.checkpoint)
            assert os.path.isfile(self.args.checkpoint)
            model_dict = self.read_checkpoint()

        # Get model
        model = self.get_model()

        # Get criterion
        criterion = self.get_criterion()

        # Get optimizer
        optimizer = self.get_optimizer(model)

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda()

        if self.args.checkpoint:
            start_iter, best_loss = self.load_checkpoint(model, optimizer, model_dict)

        # Eval only
        if bool(self.args.eval_only):
            print("Visualize Test on training set.......")
            model.eval()
            iter_loader = iter(train_loader)
            sample = next(iter_loader)
            new_loss = self.evaluate_nsteps(model, criterion, train_loader, step_id=-1, val_iters=5, vis=True)
            print("Visualize Test on test set.......")
            model.eval()
            new_loss = self.evaluate_nsteps(model, criterion, test_loader, step_id=-1, val_iters=5, vis=True)
            return model

        # Training loop
        iter_loader = iter(train_loader)
        model.train()
        pbar = trange(start_iter, self.args.train_iters)
        for step_id in pbar:
            try:
                sample = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                sample = next(iter_loader)

            loss = self.train_one_step(model, criterion, optimizer, step_id, sample)
            pbar.set_description(f"Loss: {loss:.03f}")
            if (step_id + 1) % self.args.val_freq == 0:
                print(f"step is is {step_id}")
                print("Training set evaluation.......")
                new_loss = self.evaluate_nsteps(model, criterion, train_loader, step_id, val_iters=5, split="train")
                print("Test set evaluation.......")
                new_loss = self.evaluate_nsteps(model, criterion, test_loader, step_id, val_iters=5, split="val")
                best_loss = self.save_checkpoint(model, optimizer, step_id, new_loss, best_loss)
        return model

    def train_one_step(self, model, criterion, optimizer, step_id, sample):
        """Run a single training step."""
        pass

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters, split="val"):
        """Run a given number of evaluation steps."""
        return None

    def read_checkpoint(self):
        """Load from checkpoint."""
        print("=> loading checkpoint '{}'".format(self.args.checkpoint))
        model_dict = torch.load(self.args.checkpoint, map_location="cpu")
        return model_dict

    def load_checkpoint(self, model, optimizer, model_dict, load_optimizer=True):
        model.load_state_dict(model_dict["weight"])
        if "optimizer" in model_dict and load_optimizer:
            optimizer.load_state_dict(model_dict["optimizer"])
            for p in range(len(optimizer.param_groups)):
                optimizer.param_groups[p]["lr"] = self.args.lr
        start_iter = model_dict.get("iter", 0)
        best_loss = model_dict.get("best_loss", None)

        print("=> loaded successfully '{}' (step {})".format(self.args.checkpoint, model_dict.get("iter", 0)))
        del model_dict
        torch.cuda.empty_cache()
        return start_iter, best_loss

    def save_checkpoint(self, model, optimizer, step_id, new_loss, best_loss):
        """Save checkpoint if requested."""
        if new_loss is None or best_loss is None or new_loss <= best_loss:
            best_loss = new_loss
            torch.save({
                "weight": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": step_id + 1,
                "best_loss": best_loss,
            }, self.args.log_dir / "best.pth")
        torch.save({
            "weight": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": step_id + 1,
            "best_loss": best_loss,
        }, self.args.log_dir / "last.pth")
        return best_loss


class TrainTester(BaseTrainTester):
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

    def get_datasets(self):
        # Initialize datasets with arguments
        train_dataset = KeypointDataset(
            dataset_path=self.args.dataset,
            num_iters=self.args.train_iters,
            training=True,
            interpolation_length=self.args.traj_len,
            n_kp=self.args.n_kp,
            augment_axis=[int(x) for x in self.args.augment_axis],
        )
        test_dataset = KeypointDataset(
            dataset_path=self.args.valset,
            training=False,
            interpolation_length=self.args.traj_len,
            n_kp=self.args.n_kp,
            augment_axis=[int(x) for x in self.args.augment_axis],
        )
        return train_dataset, test_dataset

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        _model = KalmDiffuser(
            unet_embedding_dim=self.args.embedding_dim,
            diffusion_timesteps=self.args.diffusion_timesteps,
            traj_len=self.args.traj_len,
            n_kp=self.args.n_kp,
        )
        return _model

    @staticmethod
    def get_criterion():
        return TrajectoryCriterion()

    def train_one_step(self, model, criterion, optimizer, step_id, sample):
        model.train()

        """Run a single training step."""
        if step_id % self.args.accumulate_grad_batches == 0:
            optimizer.zero_grad()

        device = next(model.parameters()).device
        noise_9d, model_preddict = model(
            sample["keypoints_feat"].to(device),
            sample["keypoints_xyz"].to(device),
            sample["gt_trajectory"].to(device),
        )

        # Backward pass
        loss = criterion.compute_loss(noise_9d, model_preddict, sample["gt_trajectory"].to(device))
        loss.backward()

        # Update
        if step_id % self.args.accumulate_grad_batches == self.args.accumulate_grad_batches - 1:
            optimizer.step()

        # Log
        if (step_id + 1) % self.args.val_freq == 0:  # dist.get_rank() == 0 and
            self.writer.add_scalar("lr", self.args.lr, step_id)
            self.writer.add_scalar("train-loss/noise_mse", loss, step_id)

        return loss.item()

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters, split="val", vis=False):
        """Run a given number of evaluation steps."""
        if self.args.val_iters != -1:
            val_iters = self.args.val_iters
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in enumerate(loader):
            bs = sample["keypoints_feat"].shape[0]
            if i == val_iters:
                break

            action = model(
                sample["keypoints_feat"].to(device),
                sample["keypoints_xyz"].to(device),
                None,
                run_inference=True,
            )
            metrics = criterion.compute_metrics(action, sample["gt_trajectory"].to(device))
            if vis:
                pcd_obs = sample["keypoints_xyz"].detach().cpu().numpy()
                trajs = action.detach().cpu().numpy()
                for vis_sample_i in range(1):
                    trimesh_pcdobs = trimesh.points.PointCloud(pcd_obs[vis_sample_i])
                    trimesh_pcdobs.colors = np.array([255, 0, 0])
                    trimesh_pcdtraj = trimesh.points.PointCloud(trajs[vis_sample_i][:, :3])
                    trimesh_pcdtraj.colors = np.linspace(np.array([0, 100, 0]), np.array([0, 255, 0]), num=trajs.shape[1])
                    trimesh.Scene([trimesh_pcdobs, trimesh_pcdtraj]).show()

            # Gather global statistics
            for n, l in metrics.items():
                key = f"{split}-losses/mean/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

        # Log all statistics
        values = {k: v.mean().item() for k, v in values.items()}
        if step_id > -1:
            for key, val in values.items():
                self.writer.add_scalar(key, val, step_id)

        # Also log to terminal
        print(f"Step {step_id}:")
        for key, value in values.items():
            print(f"{key}: {value:.03f}")

        return values.get("val-losses/mean/pos_l2_loss")


class TrajectoryCriterion:
    def __init__(self):
        pass

    def compute_loss(self, diffusion_noise, pred_dict, gt_trajectory):
        loss_tgt_pos = gt_trajectory[..., :3] if DIFFUSION_PREDICTION_TYPE_POS == "sample" else diffusion_noise[..., :3]
        loss_tgt_rot = gt_trajectory[..., 3:9] if DIFFUSION_PREDICTION_TYPE_ROT == "sample" else diffusion_noise[..., 3:9]
        loss_pos = 30 * F.l1_loss(pred_dict.position, loss_tgt_pos, reduction="mean")
        loss_rot = 10 * F.l1_loss(pred_dict.rotation_6d, loss_tgt_rot, reduction="mean")
        loss = loss_pos + loss_rot
        gt_openess = gt_trajectory[..., 9:]
        if torch.numel(gt_openess) > 0:
            loss_gripper = F.binary_cross_entropy_with_logits(pred_dict.openess, gt_openess)
            loss += loss_gripper
        return loss

    @staticmethod
    def compute_metrics(pred, gt):
        # (B, T, 10) pos_3d + rotation_6d + gripper_openness_1d
        pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
        rot_l2 = ((pred[..., 3:9] - gt[..., 3:9]) ** 2).sum(-1).sqrt()  # problematic but fine for now
        openess_correctness = ((pred[..., 9:] >= 0.5) == (gt[..., 9:] >= 0.5)).bool()

        # Trajectory metrics
        metrics = {
            "action_mse": F.mse_loss(pred, gt),
            "pos_l2_loss": pos_l2.mean(),
            "pos_acc": (pos_l2 < 0.01).float().mean(),
            "rot_l2_loss": rot_l2.mean(),
            "rot_acc": (rot_l2 < 0.025).float().mean(),
            "gripper_acc": openess_correctness.flatten().float().mean(),
        }

        return metrics
