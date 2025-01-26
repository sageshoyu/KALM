dataset=keypoint_files/example/kalmdiffuser_train.npz
valset=keypoint_files/example/kalmdiffuser_val.npz
lr=1e-4
batch=512

python -m scripts.main_train_kalmdiffuser \
    --dataset $dataset \
    --valset $valset \
    --num_workers 12  \
    --train_iters 60000 \
    --embedding_dim 256 \
    --diffusion_timesteps 100 \
    --val_freq 4000 \
    --batch_size $batch \
    --batch_size_val 8 \
    --lr $lr\
    --traj_len 48 \
    --n_kp 8 \
    --augment_axis '2' \
    --exp_log_dir example \
    --run_log_dir 001
