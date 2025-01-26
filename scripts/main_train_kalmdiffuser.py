import glob
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import tap
import torch

from kalm.trainer import TrainTester
from kalm.utils import save_files


class TrainerArguments(tap.Tap):
    seed: int = 0
    checkpoint: Optional[Path] = None
    accumulate_grad_batches: int = 1
    val_freq: int = 500
    eval_only: int = 0

    # Dataset
    dataset: Path = None
    valset: Path = None
    traj_len: int = 50
    augment_axis: str = "012"

    # IO
    base_log_dir: Path = Path(__file__).parent
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"

    # Main training parameters
    num_workers: int = 1
    batch_size: int = 16
    batch_size_val: int = 4
    lr: float = 1e-4
    train_iters: int = 200000
    val_iters: int = -1  # -1 means heuristically-defined

    # Model
    embedding_dim: int = 128
    n_kp: int = 8
    diffusion_timesteps: int = 100


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Arguments
    args = TrainerArguments().parse_args()
    print("Arguments:")
    print(args)
    print("-" * 100)
    if not args.eval_only:
        log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
        args.log_dir = log_dir
        log_dir.mkdir(exist_ok=True, parents=True)

        src_file_paths = glob.glob("**.py") + glob.glob("**.sh")
        src_file_paths += glob.glob(f"kalm/**.py")
        subfolders = ["hw_utils", "configs"]
        for subfolder in subfolders:
            src_file_paths += glob.glob(f"kalm/{subfolder}/**.py")
        save_files(src_file_paths, log_dir / "code")
        print("Logging:", log_dir)
        args.save(str(log_dir / "hparams.json"))

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Run
    train_tester = TrainTester(args)
    train_tester.main()
