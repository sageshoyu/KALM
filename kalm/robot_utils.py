import itertools
import math
import os.path as osp
import pickle
import zlib
from typing import Dict, List

import numpy as np
import pybullet as p
import yourdfpy
import zmq

import kalm.configs.path_config as path_config

from kalm.pybullet_utils import create_floor, LockRenderer, WorldSaver, body_collision, connect, create_sphere, get_all_links, \
    get_client, get_joint_position, get_joint_positions, get_joints, get_link_name, get_link_names, get_link_pose, get_self_link_pairs, \
    joint_from_name, link_from_name, load_pybullet, pairwise_link_collision, pose_from_tform, set_client, set_joint_position, \
    set_joint_positions, set_point, tform_from_pose, get_image

print_log = print


VERBOSE = True
FR3_CONTROLLER_ADDR = ""

DEOXYS_EE2CAM = np.array([
    [0.01019998, -0.99989995, 0.01290367, 0.03649885],
    [0.9999, 0.0103, 0.0057, -0.034889],
    [-0.00580004, 0.01280367, 0.99989995, -0.04260014],
    [0.0, 0.0, 0.0, 1.0],
])

CAM2HAND = np.array([
    [0.01021377, 0.99993106, -0.00579259, 0.03485788],
    [-0.99986457, 0.0102875, 0.01284563, 0.03608995],
    [0.01290433, 0.0056606, 0.99990071, -0.0596676],
    [0.0, 0.0, 0.0, 1.0],
])


def sample_pc(pc, num_samples=400):
    if pc.shape[0] > num_samples:
        c_mask = np.zeros(pc.shape[0], dtype=int)
        c_mask[:num_samples] = 1
        np.random.shuffle(c_mask)
        masked_pc = pc[c_mask.nonzero()]
    else:
        masked_pc = np.pad(pc, ((0, num_samples - pc.shape[0]), (0, 0)), "wrap")
    return masked_pc


class PybulletRobot:
    def __init__(self, robot_pybullet_body, camera_link_name, floor_pybullet_body):
        self.robot = robot_pybullet_body
        self.num_dof = len(get_joints(self.robot))
        self.camera_link = link_from_name(self.robot, camera_link_name)
        self.floor_body = floor_pybullet_body

    def set_joint_positions(self, positions, joints=None):
        if len(positions) == self.num_dof and joints is None:
            set_joint_positions(self.robot, range(self.num_dof), positions)
            return
        assert joints is not None
        set_joint_positions(self.robot, joints, positions)

    def get_link_pose_given_joint(self, link, positions, joints=None):
        self.set_joint_positions(positions, joints)
        return get_link_pose(self.robot, link)

    def get_link_pose(self, link):
        return get_link_pose(self.robot, link)

    def get_link_names(self):
        return get_link_names(self.robot, get_all_links(self.robot))

    def get_link_name(self, link_id):
        return get_link_name(self.robot, link_id)

    def get_joint_positions(self):
        return get_joint_positions(self.robot, get_joints(self.robot))

    def get_joint_positions_by_names(self, joint_names):
        return [get_joint_position(self.robot, self.jointid_from_name(joint_name)) for joint_name in joint_names]

    def jointid_from_name(self, joint_name):
        return joint_from_name(self.robot, joint_name)

    def get_links(self):
        return get_all_links(self.robot)

    def link_from_name(self, link_name):
        return link_from_name(self.robot, link_name)

    def get_nonadjacent_linkpairs(self):
        return get_self_link_pairs(self.robot, get_joints(self.robot))

    def get_selfcollision_linkpairs(self, exclude_linkname_pairs=None):
        if exclude_linkname_pairs is None:
            exclude_linkname_pairs = []
        exclude_link_pairs = []
        for p1, p2 in exclude_linkname_pairs:
            try:
                link1 = self.link_from_name(p1)
                link2 = self.link_from_name(p2)
                exclude_link_pairs.append((link1, link2))
            except Exception as e:
                print(f"Exclude collision links {p1} {p2} not in link names. skip")
        nonadjacent_links = self.get_nonadjacent_linkpairs()
        nonadjacent_links = list(
            filter(
                lambda pair: (pair not in exclude_link_pairs) and (pair[::-1] not in exclude_link_pairs),
                nonadjacent_links,
            )
        )
        return nonadjacent_links


# from skill_learning_for_tamp/pybullet_utils.py
def load_asset(asset_path):
    urdf_model = yourdfpy.URDF.load(
        asset_path,
        build_scene_graph=True,
        build_collision_scene_graph=True,
        load_meshes=True,
        load_collision_meshes=True,
        mesh_dir="./asset/franka_description",
        force_mesh=False,
        force_collision_mesh=False,
    )
    return urdf_model


class IKSolver_Wrapper:
    def __init__(self, ik_solver, urdf_model: yourdfpy.URDF):
        self.ik_solver = ik_solver
        self.urdf_model = urdf_model

    # from skill_learning_for_tamp/pybullet_utils.py
    def solve(self, tool_pose, seed_conf=None,
        pos_tolerance=1e-4, ori_tolerance=math.radians(5e-2)
    ):
        # will use random seed_conf if seed_conf is None
        tform = tform_from_pose(tool_pose)
        bx, by, bz = pos_tolerance * np.ones(3)
        brx, bry, brz = ori_tolerance * np.ones(3)
        conf = self.ik_solver.ik(tform, qinit=seed_conf, bx=bx, by=by, bz=bz, brx=brx, bry=bry, brz=brz)
        return conf

    def solve_closest(self, tool_pose, seed_conf, **kwargs):
        return self.solve(tool_pose, seed_conf=seed_conf, **kwargs)


class RobotController:
    def __init__(self, urdf_path, tool_link_name, use_ik=True):
        self.urdf_path = urdf_path
        if use_ik:
            self.ik_solver: IKSolver_Wrapper = self.create_ik_solver(tool_link_name)
            self.joint_names = self.ik_solver.ik_solver.joint_names
        else:
            self.ik_solver = None
            self.joint_names = None

    def create_ik_solver(self, tool_link_name):
        tracik_solver, urdf_model = self.create_tracik_solver(tool_link_name)
        ik_solver = IKSolver_Wrapper(tracik_solver, urdf_model)
        return ik_solver

    def create_tracik_solver(self, tool_link, max_time=0.0025, error=1e-2):
        urdf_path = self.urdf_path
        # TODO: unnecessary dependency
        urdf_model = load_asset(urdf_path)
        base_link_name = urdf_model._base_link
        from tracikpy import TracIKSolver

        tracik_solver = TracIKSolver(
            urdf_file=urdf_path,
            base_link=base_link_name,
            tip_link=tool_link,
            timeout=max_time,
            epsilon=error,
            solve_type="Distance",
        )  # Speed | Distance | Manipulation1 | Manipulation2
        assert tracik_solver.joint_names
        tracik_solver.urdf_file = urdf_path
        return tracik_solver, urdf_model

    def capture_image(self, *args, **kwargs):
        raise NotImplemented


class PandaPybulletController(RobotController):
    def __init__(self, robot_pb: PybulletRobot, camera_link_name, tool_link_name, urdf_path):
        super().__init__(urdf_path, tool_link_name)
        self.robot = robot_pb
        self.camera_link = link_from_name(self.robot.robot, camera_link_name)
        self.camera_intrinsics = None
        self.cam_param = None

    def get_camera_param_and_robotjoint(self):
        camera_image_size = [480, 640]
        camera_fov_w = 69.40
        camera_focal_length = (float(camera_image_size[1]) / 2) / np.tan((np.pi * camera_fov_w / 180) / 2)
        intrinsics = np.array([
            [camera_focal_length, 0, float(camera_image_size[1]) / 2],
            [0, camera_focal_length, float(camera_image_size[0]) / 2],
            [0, 0, 1],
        ])
        robotjoint = self.robot.get_joint_positions_by_names(self.joint_names)
        if self.camera_intrinsics is None:
            self.camera_intrinsics = intrinsics
        im_h, im_w = camera_image_size
        return im_h, im_w, intrinsics, robotjoint

    def capture_image(self):
        rgb_image, dep_image = get_image(self.cam_param)
        return rgb_image, dep_image, self.cam_param["camera_intr"]

    def capture_smoothed_image(self):
        return self.capture_image()

    def get_gt_segmentation(self):
        rgb_image, dep_image, mask = get_image(self.cam_param, segmentation=True)
        return rgb_image, dep_image, mask

    def command_arm(self, pos_frankainterfaceformats: Dict):
        joint_pos, joint_id = [], []
        for joint_name, joint_pos_val in pos_frankainterfaceformats.items():
            joint_pos.append(joint_pos_val)
            joint_id.append(self.robot.jointid_from_name(joint_name))
        self.robot.set_joint_positions(joint_pos, joint_id)
        for _ in range(50):
            p.stepSimulation(physicsClientId=get_client())

    def get_current_joint_confs(self):
        return self.robot.get_joint_positions_by_names(self.joint_names)

    def execute_position_path(self, pos_frankainterfaceformats: List[Dict]):
        for pos_frankainterfaceformat in pos_frankainterfaceformats:
            self.command_arm(pos_frankainterfaceformat)

    def reset_error(self):
        pass


class PandaRealworldDummyController(RobotController):
    def __init__(self, urdf_path, tool_link_name):
        super().__init__(urdf_path, tool_link_name, use_ik=False)
        self.image_rgb = None
        self.image_depth = None
        self.image_pcd = None
        self.camera_intrinsics = None
        self.camera_extrinsic = None

    def load_dummy_data(self, path):
        data = np.load(path, allow_pickle=True)

        if "rgb" in data:
            self.image_rgb = data["rgb"]
        else:
            raise ValueError("No rgb image found")

        if "extrinsic" in data:
            self.camera_extrinsic = data["extrinsic"]
        else:
            raise ValueError("No camera extrinsic found")

        if "depth" in data and "intrinsic" in data:
            self.image_depth = data["depth"]
            self.camera_intrinsics = data["intrinsic"]
        elif "pcd" in data:
            self.image_pcd = data["pcd"]
        else:
            raise ValueError("Either depth and intrinsic or pcd is needed")

    def get_camera_param_and_robotjoint(self):
        if self.camera_intrinsics is None:
            rgb_image, dep_image, intrinsics = self.capture_image()
            joint_by_name = self.get_current_joint_confs()
            im_h, im_w = dep_image.shape
            self.camera_intrinsics = intrinsics
            self.image_dim = [im_h, im_w]
            return im_h, im_w, intrinsics, joint_by_name
        else:
            joint_by_name = self.get_current_joint_confs()
            im_h, im_w = self.image_dim
            return im_h, im_w, self.camera_intrinsics, joint_by_name

    def capture_image(self):
        return self.image_rgb, self.image_depth, self.camera_intrinsics

    def capture_pcd(self):
        return self.image_rgb, self.image_pcd

    def execute_cartesian_impedance_path(self, poses, gripper_isopen, speed_factor=3):
        print(f"execute_cartesian_impedance_path: {poses} gripper_isopen={gripper_isopen} speed_factor={speed_factor}")
        return True

    def execute_joint_impedance_path(self, poses, gripper_isopen: list, speed_factor=3):
        print(f"execute_joint_impedance_path: {poses} gripper_isopen={gripper_isopen} speed_factor={speed_factor}")
        return True

    def open_gripper(self):
        print("open_gripper")
        return True

    def close_gripper(self):
        print("close_gripper")
        return True

    def get_current_joint_confs(self):
        return {"qpos": np.zeros(7), "ee_pose": np.zeros(6)}

    def go_to_home(self, gripper_open=False):
        print(f"go_to_home: gripper_open={gripper_open}")
        return True

    def free_motion(self, gripper_open=False, timeout=3.0):
        print(f"free_motion: gripper_open={gripper_open} timeout={timeout}")
        return True

    def reset_joint_to(self, qpos, gripper_open=False):
        print(f"reset_joint_to: {qpos} gripper_open={gripper_open}")
        return True


class PandaRealworldController(RobotController):
    def __init__(self, urdf_path, tool_link_name):
        super().__init__(urdf_path, tool_link_name)
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(FR3_CONTROLLER_ADDR)
        self.socket = socket
        self.camera_intrinsics = None
        self.image_dim = None
        self.capture_rs = None

    def get_camera_param_and_robotjoint(self):
        if self.camera_intrinsics is None:
            rgb_image, dep_image, intrinsics = self.capture_image()
            joint_by_name = self.get_current_joint_confs()
            im_h, im_w = dep_image.shape
            self.camera_intrinsics = intrinsics
            self.image_dim = [im_h, im_w]
            return im_h, im_w, intrinsics, joint_by_name
        else:
            joint_by_name = self.get_current_joint_confs()
            im_h, im_w = self.image_dim
            return im_h, im_w, self.camera_intrinsics, joint_by_name

    def capture_image(self):
        self.socket.send(zlib.compress(pickle.dumps({"message_name": "capture_realsense"})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))

        return (
            message["rgb"],
            message["depth"],
            message["intrinsics"],
        )  # dep in m (not mm, not need to /1000)

    def capture_smoothed_image(self, smooth_n=10, big_jump_thr=0.03):
        dep_smoothed = []
        rgb, dep, intrinsics = self.capture_image()
        dep_smoothed.append(dep)
        for _ in range(smooth_n - 1):
            depth = self.capture_rs.capture(dep_only=True)
            dep = depth / 1000.0
            assert np.isnan(dep).sum() == 0
            dep_smoothed.append(np.array(dep))

        valid_cnt = np.zeros_like(dep)

        anchor = np.median(np.stack(dep_smoothed), axis=0)
        filtered_smoothed = []
        for dep in dep_smoothed:
            dep[(anchor == 0) & (np.abs(dep - anchor) > big_jump_thr)] = 0
            filtered_smoothed.append(dep)
        smoothed = filtered_smoothed[0]
        valid_cnt[smoothed != 0] = 1
        smoothed[np.isnan(smoothed)] = 0
        for dep in filtered_smoothed[1:]:
            valid_cnt[dep != 0] += 1
            dep[np.isnan(dep)] = 0
            smoothed += dep
        valid_cnt[valid_cnt == 0] = 1
        smoothed = np.divide(smoothed, valid_cnt)
        return rgb, smoothed, intrinsics

    def execute_cartesian_impedance_path(self, poses, gripper_isopen, speed_factor=3):
        self.socket.send(zlib.compress(pickle.dumps({
            "message_name": "execute_posesmat4_osc",
            "ee_poses": poses,
            "speed_factor": speed_factor,
            "gripper_isopen": gripper_isopen,
        })))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def execute_joint_impedance_path(self, poses, gripper_isopen: list, speed_factor=3):
        self.socket.send(zlib.compress(pickle.dumps({
            "message_name": "execute_joint_impedance_path",
            "joint_confs": poses,
            "gripper_isopen": gripper_isopen,
            "speed_factor": speed_factor,
        })))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def open_gripper(self):
        self.socket.send(zlib.compress(pickle.dumps({"message_name": "open_gripper"})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def close_gripper(self):
        self.socket.send(zlib.compress(pickle.dumps({"message_name": "close_gripper"})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def get_current_joint_confs(self):
        self.socket.send(zlib.compress(pickle.dumps({"message_name": "get_joint_states"})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def go_to_home(self, gripper_open=False):
        self.socket.send(zlib.compress(pickle.dumps({"message_name": "go_to_home", "gripper_open": gripper_open})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message["success"]

    def free_motion(self, gripper_open=False, timeout=3.0):
        self.socket.send(zlib.compress(pickle.dumps({
            "message_name": "free_motion_control",
            "gripper_open": gripper_open,
            "timeout": timeout,
        })))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message["success"]

    def reset_joint_to(self, qpos, gripper_open=False):
        self.socket.send(zlib.compress(pickle.dumps({
            "message_name": "reset_joint_to",
            "gripper_open": gripper_open,
            "qpos": qpos,
        })))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message["success"]


class PandaPolicy:
    """
    Dimsam Franka Controller Interface
    """
    def __init__(self, panda_pybullet: PybulletRobot, robot_controller: RobotController):
        self.robot_pybullet = panda_pybullet
        self.robot_controller = robot_controller
        self.non_base_links = self.get_nonbase_links()
        self.collision_checker = self.create_collision_checker()

        self.anchor_obj = create_sphere(0.04, mass=0)
        set_point(self.anchor_obj, (0, -10, -20))

    def get_nonbase_links(self):
        non_base_links = self.robot_pybullet.get_links()
        if self.robot_controller.ik_solver is not None:
            non_base_links.remove(self.robot_pybullet.link_from_name(self.robot_controller.ik_solver.urdf_model._base_link))
        else:
            print("No ik solver")
        return non_base_links

    def get_camera_param(self):
        im_h, im_w, intrinsics, robot_joint_positions = self.robot_controller.get_camera_param_and_robotjoint()
        campose_from_joint = self.robot_pybullet.get_link_pose_given_joint(
            self.robot_pybullet.camera_link,
            robot_joint_positions,
            [self.robot_pybullet.jointid_from_name(jn) for jn in self.robot_controller.joint_names],
        )
        camera_pose = tform_from_pose(campose_from_joint)

        # some from https://github.com/columbia-ai-robotics/dsr/blob/f34d60f885cd01e6b562e799d7c81eafda3ae765/sim.py
        opengl_campose = camera_pose.copy()
        opengl_campose[:, 1:3] = -opengl_campose[:, 1:3]
        camera_view_matrix = (np.linalg.inv(opengl_campose).T).reshape(-1)

        camera_z_near = 0.01
        camera_z_far = 10.0
        camera_image_size = [im_h, im_w]
        camera_fov_h = (math.atan((float(camera_image_size[0]) / 2) / intrinsics[0][0]) * 2 / np.pi) * 180
        camera_projection_matrix = p.computeProjectionMatrixFOV(
            fov=camera_fov_h,
            aspect=float(camera_image_size[1]) / float(camera_image_size[0]),
            nearVal=camera_z_near,
            farVal=camera_z_far,
            physicsClientId=get_client(),
        )  # notes: 1) FOV is vertical FOV 2) aspect must be float
        camera_param = {
            "camera_image_size": camera_image_size,
            "camera_intr": intrinsics,  # if intrinsics is not None else camera_intrinsics,
            "camera_pose": camera_pose,
            "camera_view_matrix": camera_view_matrix,
            "camera_projection_matrix": camera_projection_matrix,
            "camera_z_near": camera_z_near,
            "camera_z_far": camera_z_far,
        }

        if isinstance(self.robot_controller, PandaPybulletController) and self.robot_controller.cam_param is None:
            self.robot_controller.cam_param = camera_param

        return camera_param

    def capture_image(self, smooth=False):
        if not smooth:
            return self.robot_controller.capture_image()
        return self.robot_controller.capture_smoothed_image()

    def create_collision_checker(self, collision_checking_pointcloudsample=300):
        arm_in_pb = self.robot_pybullet.robot  # robot in sim
        non_base_arm_links = self.non_base_links
        floor_in_pb = self.robot_pybullet.floor_body
        non_adjacent_pairs = self.robot_pybullet.get_selfcollision_linkpairs(
            exclude_linkname_pairs=[
                ["panda_link7", "panda_hand"],
                ["panda_leftfinger", "panda_rightfinger"],
                ["ruler", "panda_leftfinger"],
                ["ruler", "panda_rightfinger"],
                ["ruler", "panda_hand"],
                ["panda_leftfinger", "camera_color_optical_frame"],
                ["panda_rightfinger", "camera_color_optical_frame"],
                ["panda_link7", "panda_link5"],
                ["panda_link7", "ruler"],
                ["camera_color_optical_frame", "ruler"],
            ]
        )
        obstacle_cloud_pyobj = create_sphere(0.04, mass=0)
        disable_point = (0, -10, -10)
        set_point(obstacle_cloud_pyobj, disable_point)

        def check_collision(conf, scene_pointcloud_worldframe=None):
            try:
                with WorldSaver():
                    with LockRenderer():
                        for joint_i, joint_val_i in enumerate(conf):
                            set_joint_position(arm_in_pb, joint_i, joint_val_i)
                        if any(pairwise_link_collision(arm_in_pb, link, floor_in_pb, max_distance=3e-2) for link in non_base_arm_links):
                            if VERBOSE:
                                print_log(f"colli floor")
                            return True

                        # self collision
                        for link1, link2 in non_adjacent_pairs:
                            if pairwise_link_collision(arm_in_pb, link1, arm_in_pb, link2, max_distance=2e-2):
                                if VERBOSE:
                                    print_log(f"{self.robot_pybullet.get_link_name(link1)} {self.robot_pybullet.get_link_name(link2)} collide")
                                return True

                        if scene_pointcloud_worldframe is not None:
                            cloud_nonfloor = scene_pointcloud_worldframe[scene_pointcloud_worldframe[..., 2] > 0.003]
                            sample_points_collisioncheck = sample_pc(cloud_nonfloor, collision_checking_pointcloudsample)
                            for pt in sample_points_collisioncheck:
                                set_point(obstacle_cloud_pyobj, pt)
                                if body_collision(obstacle_cloud_pyobj, arm_in_pb):
                                    set_point(obstacle_cloud_pyobj, disable_point)
                                    print_log(f"collide cloud")
                                    return True
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Collision. Error: {e}")
            finally:
                set_point(obstacle_cloud_pyobj, disable_point)
            return False

        return check_collision

    def plan_workspace_motion(
        self, waypoint_poses7d, waypoint_grippers, obstacle_pointcloud,
        start_conf=None, trim_traj=True,
    ):
        if start_conf is None:
            start_conf = self.robot_controller.get_current_joint_confs()["qpos"][:7]
        ik_solutions = self.compute_arm_ik_waypoints(waypoint_poses7d, start_conf=start_conf)
        if ik_solutions is None:
            # No IK solution for one or more intermediate steps
            print_log("plan_arm_traj ik fail")
            return None, None, None

        arm_path, gripper_path = self.plan_arm_path_given_conf(ik_solutions, waypoint_grippers, obstacle_pointcloud, trim_traj=trim_traj)
        return ik_solutions, arm_path, gripper_path

    def compute_arm_ik_waypoints(self, tool_waypoint_poses, start_conf, obstacle_pointcloud=None):
        ik_solutions = [start_conf]
        for i, tgt_pose in enumerate(tool_waypoint_poses):
            ik_solution_stepi_candidates = [self.robot_controller.ik_solver.solve(tgt_pose)] * 45
            ik_solution_stepi_candidates += [self.robot_controller.ik_solver.solve_closest(tgt_pose, seed_conf=ik_solutions[-1])] * 35

            valid_candidates = [sol_i for sol_i in ik_solution_stepi_candidates if sol_i is not None]
            if len(valid_candidates) == 0:
                if VERBOSE:
                    print_log(f"Tgt conf not reachable.")
                return None
            valid_candidates = [sol_i for sol_i in valid_candidates if not self.collision_checker(sol_i, obstacle_pointcloud)]
            if len(valid_candidates) == 0:
                if VERBOSE:
                    print_log(f"Tgt conf all in collision.")
                return None
            if len(valid_candidates) == 0:
                if len(ik_solutions) == 0:
                    if VERBOSE:
                        print_log(f"IK for first step failed")
                    return None
                if VERBOSE:
                    print_log(f"IK for intermediate step {i} failed")
                return None
            else:
                ik_solution_stepi_candidates_dist = [np.linalg.norm(sol_i - ik_solutions[-1]) for sol_i in valid_candidates]
                closest_solution_id = np.argmin(ik_solution_stepi_candidates_dist, axis=0)
                ik_solution_stepi = valid_candidates[closest_solution_id]
            ik_solutions.append(ik_solution_stepi)
        return ik_solutions

    def get_interpolated_path(self, conf1, conf2, collision_fn, extend_fn):
        interpolated_path = [conf1]
        for conf in list(extend_fn(conf1, conf2)):
            if collision_fn(conf):
                return None
            interpolated_path.append(conf)
        return interpolated_path

    def plan_arm_path_given_conf(self, confs, grippers, obstaclecloud=None, trim_traj=True):
        try:
            from pybullet_planning.motion_planners.rrt_connect import birrt  # https://github.com/caelan/pybullet-planning
        except ImportError:
            raise ImportError("Please install pybullet-planning (https://github.com/caelan/pybullet-planning)")

        lower, upper = self.robot_controller.ik_solver.ik_solver.joint_limits
        jointlimit_scale = upper - lower

        def arm_rrt_sample_fn():
            scaled = np.random.rand(len(confs[0])) * jointlimit_scale
            return lower + scaled

        def arm_rrt_distance_fn(q1, q2):
            return np.linalg.norm(np.array(q1) - np.array(q2))

        def arm_rrt_extend_fn(q1, q2):
            q1, q2 = np.array(q1), np.array(q2)
            resolutions_rel100 = 0.05
            resolutions = resolutions_rel100 * jointlimit_scale
            steps = int(max(np.divide(np.abs(q1 - q2), resolutions)))
            if steps == 0:
                steps = 1
            step_vec = (q2 - q1) / steps
            extended_traj = []
            for i in range(steps):
                extended_traj.append(q1 + step_vec * i)
            extended_traj.append(q2)
            return extended_traj

        arm_rrt_colli_fn_withcloud = lambda q: self.collision_checker(q, obstaclecloud)
        arm_rrt_colli_fn_withoutcloud = lambda q: self.collision_checker(q)

        arm_path = []
        grippers_path = []
        for i in range(len(confs) - 1):
            cur_step_colli_fn = arm_rrt_colli_fn_withcloud

            interpolated_path = self.get_interpolated_path(confs[i], confs[i + 1], cur_step_colli_fn, arm_rrt_extend_fn)
            if interpolated_path is not None:
                arm_path.append(interpolated_path)
                grippers_path.append([grippers[i] for _ in range(len(interpolated_path))])
                continue
            arm_path_step = birrt(
                confs[i],
                confs[i + 1],
                arm_rrt_distance_fn,
                arm_rrt_sample_fn,
                arm_rrt_extend_fn,
                collision_fn=cur_step_colli_fn,
                max_time=20,
                max_iterations=2000,
            )
            arm_path_step_smoothed = arm_path_step
            if arm_path_step_smoothed is not None:
                arm_path.append(arm_path_step_smoothed)
                grippers_path.append([grippers[i] for _ in range(len(arm_path_step_smoothed))])
            else:
                print_log(f"no rrt plan for armconf {i} to {i + 1}")
                return None, None
        if arm_path[0] is None:
            return None, None
        arm_path_flat = list(itertools.chain(*arm_path))
        grippers_path_flat = list(itertools.chain(*grippers_path))
        if trim_traj:
            arm_path_flat_trimmed = [arm_path_flat[0]]
            grippers_path_flat_trimmed = [grippers_path_flat[0]]
            for conf, grip in zip(arm_path_flat[1:], grippers_path_flat[1:]):
                if np.linalg.norm(conf - arm_path_flat_trimmed[-1]) > 0.05:
                    arm_path_flat_trimmed.append(conf)
                    grippers_path_flat_trimmed.append(grip)
        else:
            arm_path_flat_trimmed = arm_path_flat
            grippers_path_flat_trimmed = grippers_path_flat
        return arm_path_flat_trimmed, grippers_path_flat_trimmed

    def compute_arm_path_to_trajee0(
        self, target_ee_poses_mat4: list, target_grippers: list, obstacle_cloud_in_world_frame,
        start_conf=None, dense_interpolate_fac=1, trim_traj=True,
    ):
        target_panda_hand_poses_mat4 = []
        for target_ee_poses_mat4 in target_ee_poses_mat4:
            target_panda_hand_pose_mat4 = target_ee_poses_mat4.dot(DEOXYS_EE2CAM.dot(CAM2HAND))
            target_panda_hand_poses_mat4.append(pose_from_tform(target_panda_hand_pose_mat4))
        ik_solutions, arm_path, gripper_path = self.plan_workspace_motion(
            target_panda_hand_poses_mat4,
            target_grippers,
            obstacle_cloud_in_world_frame,
            start_conf=start_conf,
            trim_traj=trim_traj,
        )
        if arm_path is None:
            return None, None, None
        arm_path = np.array(arm_path)
        gripper_path = np.array(gripper_path)
        if dense_interpolate_fac:
            x_ori = np.linspace(0, 1, num=len(arm_path))
            x_new = np.linspace(0, 1, num=len(arm_path) * dense_interpolate_fac)
            arm_path_new = np.zeros((len(arm_path) * dense_interpolate_fac, arm_path.shape[1]))
            gripper_path_new = np.zeros(len(gripper_path) * dense_interpolate_fac)
            for arm_dim in range(arm_path.shape[1]):
                arm_path_new[:, arm_dim] = np.interp(x_new, x_ori, arm_path[:, arm_dim])
            gripper_path_new = np.interp(x_new, x_ori, gripper_path) > 0.5
            arm_path = arm_path_new
            gripper_path = gripper_path_new
        return ik_solutions[-1], arm_path, gripper_path

    def reset_error(self):
        self.robot_controller.reset_error()


def setup_panda_pybullet(panda_path, dummy=False):
    print(panda_path)
    floor_in_pb = create_floor()
    robot_body_pybullet = load_pybullet(panda_path, fixed_base=True)
    robot_pybullet = PybulletRobot(
        robot_body_pybullet,
        camera_link_name="camera_color_optical_frame",
        floor_pybullet_body=floor_in_pb,
    )
    if dummy:
        robot_controller = PandaRealworldDummyController(urdf_path=panda_path, tool_link_name="panda_hand")
    else:
        robot_controller = PandaRealworldController(urdf_path=panda_path, tool_link_name="panda_hand")
    policy = PandaPolicy(robot_pybullet, robot_controller)
    return policy


def initialize_robot_policy(debug=False, dummy=False):
    sim_id = connect(use_gui=debug)
    set_client(sim_id)
    panda_path = "panda_arm_hand_cam.urdf"
    panda_path = osp.join(path_config.KALM_ASSETS_ROOT, panda_path)
    robot_policy = setup_panda_pybullet(panda_path, dummy=dummy)
    p.setRealTimeSimulation(False, physicsClientId=sim_id)
    return robot_policy
