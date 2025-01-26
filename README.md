# KALM: Keypoint Abstraction using Large Models for Object-Relative Imitation Learning

**[KALM: Keypoint Abstraction using Large Models for Object-Relative Imitation Learning](https://kalm-il.github.io/)**
<br />
[Xiaolin Fang*](https://fang-xiaolin.github.io),
[Bo-Ruei Huang*](https://borueihuang.com),
[Jiayuan Mao*](https://jiayuanm.com),
[Jasmine Shone](https://jasshone.github.io),
[Joshua B. Tenenbaum](https://cocosci.mit.edu/josh),
[Tomás Lozano-Pérez](https://people.csail.mit.edu/tlp/),
[Leslie Pack Kaelbling](https://people.csail.mit.edu/lpk/)
<br />
CoRL Workshop on Language and Robot Learning, 2024. <span style="color:#CC3333">Best Paper Award</span>
<br />
[[Paper]](http://arxiv.org/abs/2410.23254)
[[Website]](https://kalm-il.github.io/)

![teaser](https://kalm-il.github.io/static/images/framework.jpeg)

```
@article{fang2024kalm,
  title={Keypoint Abstraction using Large Models for Object-Relative Imitation Learning},
  author={Xiaolin Fang and Bo-Ruei Huang and Jiayuan Mao and Jasmine Shone and Joshua B. Tenenbaum and Tomás Lozano-Pérez and Leslie Pack Kaelbling},
  journal={arXiv:2410.23254},
  year={2024}
}
```

### Installation

Install the dependencies using conda:

```bash
conda env create -f environment.yaml
conda activate kalm
pip install git+https://github.com/mhamilton723/FeatUp
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth # download the SAM model
```

Fill in the OpenAI API key in the config.

```bash
cp kalm/configs/local_config_template.py kalm/configs/local_config.py
```

### Usage

#### Keypoint Distillation

You can specify the prompt (task name and description) in `TASKNAME2DESC` in `kalm/vlm_client.py`. We provide an example for data spec.

```bash
python -m scripts.main_kalm_distill_keypoints --save_path keypoint_files/example  --task_name drawer  --data_path keypoint_files/drawer_example_traj.npz
```

Note: The example data may not pass the verification check. You can use your own data.

##### Data Format

The data is stored in a `.npz` file with the following format:

```python
{
    'reference_video_rgb': np.array,  # (N, 512, 512, 3)
    'reference_video_pcd': np.array,  # (N, 256, 256, 3)
    'verification_dataset': [
        {
            'ee_poses_worldframe_trajectory': np.array,  # (T, 4, 4)
            'gripper_openness_trajectory': np.array,  # (T, 1)
            'joint_q_trajectory': np.array,  # (T, 7)
            'extrinsic': np.array,  # (4, 4)
            'observed_rgb': np.array,  # (512, 512, 3)
            'observed_pcd': np.array,  # (256, 256, 3)
        } * N
    ]
}
```

#### Training

Train the keypoint-conditioned policy. The parameters could be found in the script.

```bash
bash scripts/train_kp_traj.sh
```

#### Real Robot Evaluation

Run the real robot evaluation. You can add your own task in `configs/model_config.py`.

```bash
python -m scripts.main_kalm_eval_robot --task drawer
```