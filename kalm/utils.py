from __future__ import annotations

import io
import os
import pickle
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import seaborn as sns
import torch


@dataclass
class GPT_CLIENT_INFO:
    gpt_client: "GPT_CLIENT"
    previous_messages: dict
    ref_image_with_masks: np.ndarray
    sequence: list
    task_desc: str
    grid_shape: tuple
    query_image: Optional[np.ndarray] = None

    def update_query_image(self, query_image):
        self.query_image = query_image


def smooth_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_selected = cv2.morphologyEx(mask.astype("uint8"), cv2.MORPH_OPEN, kernel)
    return mask_selected.astype(bool)


def sample_farthest_points_from_selected_masked_area(
    image_rgb: np.ndarray,
    masked_area: np.ndarray,
    pcd: np.ndarray,
    num_candidate_points: int = 15,
):
    """

    Args:
        image_rgb:      H W 3 numpy array. 0 - 255
        masked_area:    H W bool mask

    Returns:
        sampled_edge_points_yx:     ij on the image
    """
    image_rgb = image_rgb.copy()
    im_h, im_w = masked_area.shape
    mask_selected = smooth_mask(masked_area)
    image_rgb[~mask_selected] = 0

    # 3D fps
    pcd_masked = pcd[mask_selected]  # N x 3
    pcd_valid_masked = pcd_masked[~(pcd_masked == 0).all(axis=1)]
    sampled_edge_points_xyz = farthest_point_sampling(pcd_valid_masked, num_candidate_points)
    # take neighboring points
    pcd_close_to_selected_mask = (np.linalg.norm(sampled_edge_points_xyz - pcd.reshape((-1, 1, 3)), axis=2).min(axis=1)).reshape(im_h, im_w) < 1e-4
    sampled_edge_points_yx = np.array(np.where(pcd_close_to_selected_mask)).T
    sampled_edge_points_yx = sample_pc(
        sampled_edge_points_yx,
        num_candidate_points,
        with_replacement=sampled_edge_points_yx.shape[0] < num_candidate_points,
    )
    return sampled_edge_points_yx


def point_cloud_from_depth_image_camera_frame(depth_image, camera_intrinsics, remove_invalid_points=False):
    """
    Project depth image back to 3D to obtain partial point cloud.
    """
    height, width = depth_image.shape
    xmap, ymap = np.meshgrid(np.arange(width), np.arange(height))
    homogenous_coord = np.concatenate((xmap.reshape(1, -1), ymap.reshape(1, -1), np.ones((1, height * width))))
    rays = np.linalg.inv(camera_intrinsics).dot(homogenous_coord)
    point_cloud = depth_image.reshape(1, height * width) * rays
    point_cloud = point_cloud.transpose(1, 0).reshape(-1, 3)
    if remove_invalid_points:
        point_cloud = point_cloud[(point_cloud != 0).any(axis=1)]
    return point_cloud


def get_pcd(dep, intrinsic, resize=None, far=1.5, near=0.1):
    if dep.max() > 100:
        dep = dep / 1000.0
    pcd = point_cloud_from_depth_image_camera_frame(dep, intrinsic)
    h, w = dep.shape
    pcd_hw3 = pcd.reshape(h, w, 3)
    pcd_hw3[dep > far] = 0
    pcd_hw3[dep < near] = 0
    pcd_hw3[dep == 0] = 0
    if resize is not None:
        # do center crop
        assert h < w
        resize_w, resize_h = int(resize * w / h), resize
        pcd_hw3 = cv2.resize(pcd_hw3, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
        crop_top = 0
        crop_left = int(round((resize_w - resize) / 2.0))
        pcd_hw3 = pcd_hw3[crop_top : crop_top + resize, crop_left : crop_left + resize]
    return pcd_hw3


def np_to_o3d(array):
    if isinstance(array, o3d.geometry.PointCloud):
        return array
    assert array.shape[-1] == 3
    if len(array.shape) == 3:
        array = array.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(array))
    return pcd


def transform_pointcloud(transform_mat, input_pointcloud, set_invalid=False):
    """
    Transform a point cloud with tform mat
    Args:
        transform_mat:      4 x 4
        input_pointcloud:   N x 3

    Returns:
        transformed point cloud
    """
    original_pointcloud_hom = np.concatenate((input_pointcloud, np.ones((input_pointcloud.shape[0], 1))), axis=1)
    transformed_pointcloud = transform_mat.dot(original_pointcloud_hom.T)[:3].T
    if set_invalid:
        transformed_pointcloud[(input_pointcloud == 0).all(axis=1)] = 0
    return transformed_pointcloud


def center_crop(rgb_im):
    im_h, im_w = rgb_im.shape[:2]
    min_hw = min(im_h, im_w)
    cropped_image = rgb_im[(im_h - min_hw) // 2: (im_h - min_hw) // 2 + min_hw, (im_w - min_hw) // 2: (im_w - min_hw) // 2 + min_hw]
    return cropped_image


def get_pointcloud(
    dep_im,
    intrinsic,
    cam_pose_in_worldframe,
    frame="world",
    resize=None,
    far=1.5,
    near=0.1,
):
    assert frame in ["world", "camera"]
    original_pointcloud_camframe = get_pcd(dep_im, intrinsic, resize=resize, near=near, far=far)
    pointcloud_nonzero = original_pointcloud_camframe.reshape(-1, 3)
    pointcloud_nonzero = pointcloud_nonzero[(pointcloud_nonzero != 0).any(axis=1)]
    if frame == "camera":
        return pointcloud_nonzero
    # else return world frame
    return transform_pointcloud(cam_pose_in_worldframe, pointcloud_nonzero)


def approx_equal(arr1, arr2, thr=0.7) -> bool:
    """
    Return True if IoU of two masks > thr
    """
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same shape")
    union = np.logical_or(arr1, arr2)
    intersection = np.logical_and(arr1, arr2)
    if union.sum() == 0:
        return False
    return intersection.sum() / union.sum() > thr


# remove small regions. modified from SAM `https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py#L267C5-L267C25`
def remove_trivial_regions(mask: np.ndarray, is_nontrivial, mode: str) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    If using 'holes' mode, will fill in the small holes and add them to `mask`.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)

    sizes = stats[:, -1][1:]  # Row 0 is background label
    # small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    small_regions = [i + 1 for i, s in enumerate(sizes) if (not is_nontrivial(regions == i + 1))]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def is_degenerated_pointcloud(pointcloud_raw, min_edge_len_threshold=0.015):
    if pointcloud_raw is None:
        return False
    if len(pointcloud_raw) < 30:
        return True
    # look at only valid points (depth!=nan)
    pointcloud = pointcloud_raw[(pointcloud_raw != 0).all(axis=1)]

    if len(pointcloud) < 30:
        return True

    # check aabb
    aabb = [pointcloud.min(axis=0), pointcloud.max(axis=0)]
    aabb_len_of_each_edge = [(aabb[1][i] - aabb[0][i]) for i in range(3)]
    if min(aabb_len_of_each_edge) < min_edge_len_threshold:
        return True

    pcd = np_to_o3d(pointcloud)
    try:
        oriented_bounding_box = pcd.get_oriented_bounding_box()
        oobb_extent = oriented_bounding_box.extent
    except Exception as e:
        from pdb import traceback

        traceback.print_exc()
        print(f"Error in getting OBB")
        return True
    if min(oobb_extent) < min_edge_len_threshold:
        return True
    return False


def is_degenerated_mask(
    mask: np.ndarray,
    pointcloud=None,
    min_area_percentage_threshold=0.0001,
    narrow_area_threshold=10,
) -> Tuple[Union[np.ndarray, None], bool]:
    """
    Determine if a mask is degenerated. If not, fill in the small holes and remove flying pixels.
    """
    im_h, im_w = mask.shape[:2]
    min_area_threshold = int(min_area_percentage_threshold * im_h * im_w)

    def is_valid(msk, area_thres, bbox_len_thres):
        bbox_y, bbox_x = np.where(msk > 0)
        if not msk.sum() >= area_thres:
            return False
        is_narrow = bbox_y.max() - bbox_y.min() < bbox_len_thres or bbox_x.max() - bbox_x.min() < bbox_len_thres
        return not is_narrow

    if not is_valid(mask, min_area_threshold, narrow_area_threshold):
        return np.zeros_like(mask).astype(bool), True

    mask, is_modified = remove_trivial_regions(
        mask,
        partial(
            is_valid,
            area_thres=min_area_threshold * 1.5,
            bbox_len_thres=narrow_area_threshold,
        ),
        mode="holes",
    )
    if not is_valid(mask, min_area_threshold, narrow_area_threshold):
        return np.zeros_like(mask).astype(bool), True
    mask, is_modified = remove_trivial_regions(
        mask,
        partial(
            is_valid,
            area_thres=min_area_threshold,
            bbox_len_thres=narrow_area_threshold,
        ),
        mode="islands",
    )
    if not is_valid(mask, min_area_threshold, narrow_area_threshold):
        return np.zeros_like(mask).astype(bool), True
    if pointcloud is not None and is_degenerated_pointcloud(pointcloud[mask]):
        return None, True
    return mask, False


def resize_to(img, target_max_dim):
    max_dim = max(img.shape[:2])
    scale = target_max_dim / max_dim
    target_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    return cv2.resize(img, target_size)


def draw_grid(img, nr_vertical, nr_horizontal, resize_to_max_dim: int = 0):
    """Draw a grid on the image with nr_vertical and nr_horizontal lines. It will also put a number at the top-left corner of each cell."""
    img = img.copy()
    if resize_to_max_dim > 0:
        img = resize_to(img, resize_to_max_dim)

    h, w = img.shape[:2]
    for i in range(1, nr_vertical):
        x = i * w // nr_vertical
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 3)
    for i in range(1, nr_horizontal):
        y = i * h // nr_horizontal
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 3)

    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2
    font_thickness = 3
    text_color = (0, 0, 0)
    text_color_bg = (255, 255, 255)
    text_pad = 5

    for i in range(nr_horizontal):
        for j in range(nr_vertical):
            x = j * w // nr_vertical + w // nr_vertical // 2
            y = i * h // nr_horizontal + h // nr_horizontal // 2
            text = f"{i * nr_vertical + j + 1}"
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_w, text_h = text_size
            cv2.rectangle(img, (x, y), (x + text_w, y + text_h), text_color_bg, -1)
            cv2.putText(
                img,
                text,
                (x, y + text_h + font_scale - 1),
                font,
                font_scale,
                text_color,
                font_thickness,
            )

    return img


def extract_tag_content(content, tag):
    """
    Extract the content of a tag from a string.
    Args:
        content:    str. The content string
        tag:        str. The tag to extract

    Returns:
        str. The content of the tag.
    """
    start = content.find(f"<{tag}>")
    end = content.find(f"</{tag}>")
    if start == -1 or end == -1:
        return None
    return content[start + len(tag) + 2 : end]


def calc_grid_cell_tlbr(image_shape: tuple[int, int], grid_shape: tuple[int, int], index: int) -> tuple[int, int, int, int]:
    """
    Generate top-left and bottom-right coordinate of a grid cell.
    Args:
        image_shape:    H, W
        grid_shape:     nr_vertical, nr_horizontal
        index:          0-indexed grid cell index. The first row is 0, 1, 2, ... from left to right.

    Returns:
        top, left, bottom, right
    """

    h, w = image_shape
    grid_h, grid_w = grid_shape
    cell_h, cell_w = h / grid_h, w / grid_w
    row = index // grid_w
    col = index % grid_w
    top = int(row * cell_h)
    left = int(col * cell_w)
    bottom = int((row + 1) * cell_h)
    right = int((col + 1) * cell_w)
    return top, left, bottom, right


def draw_text(
    img,
    text,
    font=cv2.FONT_HERSHEY_PLAIN,
    pos=(0, 0),
    font_scale=3,
    font_thickness=2,
    text_color=(0, 255, 0),
    text_color_bg=(0, 0, 0),
):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(
        img,
        text,
        (x, y + text_h + font_scale - 1),
        font,
        font_scale,
        text_color,
        font_thickness,
    )
    return text_size


def draw_seg_on_im(im, pred_masks: list[np.ndarray], alpha=0.5, plot_anno=None, cm=None):
    """
    Args:
        im:     HxWx3 image array
        pred_masks:   list of HxW binary mask
        alpha:

    Returns:
        im with masks overlayed.
    """
    im = im.copy()
    n_colors = len(pred_masks)
    if cm is None:
        cm = sns.color_palette("bright", n_colors=n_colors)
    all_contours = []
    for i, obj_mask in enumerate(pred_masks):
        contours, hier = cv2.findContours(obj_mask.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours = [x for x in contours if cv2.contourArea(x) > 50]
        if len(contours) == 0:
            all_contours.append(None)
            continue
        all_contours.append(contours[0])
        contour_colors = [tuple(map(int, np.array(cm[i][:3]) * 255))] * len(contours)
        for contour_color, contour in zip(contour_colors, contours):
            cv2.drawContours(im, contour, -1, contour_color, thickness=4)
        im = im.astype(np.float32)
        im[obj_mask > 0] = im[obj_mask > 0] * alpha + np.array(np.array(cm[0][:3])) / np.array(np.array(cm[0][:3])).max() * (1 - alpha) * 255
        im = im.astype(np.uint8)
    if plot_anno is not None:
        for i, contour in enumerate(all_contours):
            if contour is None:
                continue
            M = cv2.moments(contour)
            color = tuple(map(int, np.array(cm[i][:3]) * 255))
            pos = (round(M["m10"] / M["m00"]), round(M["m01"] / M["m00"]))
            draw_text(
                im,
                str(plot_anno[i][0]),
                pos=pos,
                font_scale=4,
                font_thickness=4,
                text_color=(255, 255, 255),
                text_color_bg=color,
            )
    return im


def farthest_point_sampling(points, num_samples, return_idx=False):
    # Select the first point randomly
    idx_sel = np.random.randint(len(points))
    sampled_points = [points[idx_sel]]
    sampled_idx = [idx_sel]

    for _ in range(num_samples - 1):
        # Calculate the distance of each point to the closest selected point
        dist_to_closest = np.min(
            np.linalg.norm(points - np.array(sampled_points)[:, np.newaxis], axis=2),
            axis=0,
        )

        # Select the farthest point
        farthest_idx = np.argmax(dist_to_closest)
        sampled_points.append(points[farthest_idx])
        sampled_idx.append(farthest_idx)

    if return_idx:
        return np.array(sampled_points), np.array(sampled_idx)

    return np.array(sampled_points)


def plot_keypoints(img, vertices_yx: list[list], label_start_id=0, all_label_num=None):
    """Draw a grid on the image with nr_vertical and nr_horizontal lines. It will also put a number at the top-left corner of each cell."""
    img = img.copy()

    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 0)
    text_color_bg = (255, 255, 255)
    colormap = plt.get_cmap("gist_rainbow")
    colormap_howmany = len(vertices_yx) if all_label_num is None else all_label_num
    shift_x, shift_y = 5, 5  # don't sit on the point
    for pt_i, (y, x) in enumerate(vertices_yx):
        text = f"{label_start_id+pt_i}"
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        # color_pt_i = (np.array(colormap(pt_i+label_start_id/colormap_howmany))[:3]*255).tolist()
        color_pt_i = (0, 255, 0)
        cv2.rectangle(
            img,
            (x + shift_x, y + shift_x),
            (x + text_w + shift_x, y + text_h + shift_x),
            text_color_bg,
            -1,
        )
        cv2.putText(
            img,
            text,
            (x + shift_x, y + text_h + font_scale - 1 + shift_x),
            font,
            font_scale,
            color_pt_i,
            font_thickness,
        )
        cv2.circle(img, (x, y), radius=2, thickness=2, color=color_pt_i)
    return img


def plot_points(
    im1,
    points_plot1_xy,
    im2=None,
    points_plot2_xy=None,
    save_name_prefix=None,
    vis=True,
):
    if im2 is None:
        im2 = im1
        points_plot2_xy = points_plot1_xy
    im1 = im1.astype(np.float32) / 255.0
    im2 = im2.astype(np.float32) / 255.0
    assert len(points_plot1_xy) == len(points_plot2_xy)
    plt.clf()
    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(im1)
    ax[0].set_title(f"Ref Image")
    ax[1].imshow(im2)
    ax[1].set_title(f"Query Image")
    color_map = plt.get_cmap("gist_rainbow")
    colors = [color_map((i + 1) / (len(points_plot1_xy) + 1)) for i in range(len(points_plot1_xy))]
    ax[0].scatter(
        [pt[0] for pt in points_plot1_xy],
        [pt[1] for pt in points_plot1_xy],
        c=colors,
        s=20,
    )
    ax[1].scatter(
        [pt[0] for pt in points_plot2_xy],
        [pt[1] for pt in points_plot2_xy],
        c=colors,
        s=20,
    )
    if save_name_prefix is not None:
        ax0_vis = save_ax_nosave(ax[0])
        ax0_savename = save_name_prefix + "_ref.png"
        cv2.imwrite(ax0_savename, (ax0_vis[..., :3][..., ::-1] * 255).astype(np.uint8))
        ax1_vis = save_ax_nosave(ax[1])
        ax1_savename = save_name_prefix + "_query.png"
        cv2.imwrite(ax1_savename, (ax1_vis[..., :3][..., ::-1] * 255).astype(np.uint8))
    if vis:
        plt.show()
    plt.close()
    return


def save_ax_nosave(ax):
    ax.axis("off")
    ax.figure.canvas.draw()
    trans = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.bbox.transformed(trans)
    buff = io.BytesIO()
    plt.savefig(buff, format="png", dpi=ax.figure.dpi, bbox_inches=bbox)
    ax.axis("on")
    buff.seek(0)
    im = plt.imread(buff)
    return np.array(im)


def convert_scale1_to_scale2(pt_x, pt_y, scale1, scale2):
    return round(pt_x / scale1 * scale2), round(pt_y / scale1 * scale2)


def save_file(src_file, save_dir):
    tgt_path = os.path.join(save_dir, src_file.replace("/", "-"))
    os.system(f"cp {src_file} {tgt_path}")


def save_files(src_files, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    for src_file in src_files:
        save_file(src_file, save_dir)


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout to devnull"""
    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull) as out, redirect_stderr(fnull) as err:
            yield (out, err)


def normalize_to_01scale(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)


def sample_pc(pc, num_samples=400, with_replacement=False):
    return pc[np.random.choice(pc.shape[0], num_samples, replace=with_replacement)]


def preprocess_rgb(rgb_ims, input_size: int = 256) -> np.ndarray:
    """
    Preprocess RGB images to be used in the model.
    Args:
        rgb_ims: RGB NHWC numpy image 0-255
        input_size:
        return_normalized:

    Returns:
        NHWC numpy images
    """
    if rgb_ims.max() <= 1.0:
        print(f"Input should be in range 0-255, RGB order. Please double check the input.")
        rgb_ims = (rgb_ims * 255).astype(np.uint8)
    if rgb_ims[0].shape[0] != rgb_ims[0].shape[1]:
        # for simplicity do center crop for now
        raise NotImplementedError(f"Input should be square. Please double check the input.")
    return np.asarray([cv2.resize(rgb_im, (input_size, input_size)) for rgb_im in rgb_ims])


def preprocess_pcd(pointclouds: list[np.ndarray], input_size: int = 256) -> list[np.ndarray]:
    """
    Preprocess point cloud to be used in the model.
    Args:
        pointclouds: point cloud in camera frame. N x H x W x 3
        input_size:

    Returns:
        NHWC numpy images
    """
    if pointclouds[0].shape[0] != pointclouds[0].shape[1]:
        # for simplicity do center crop for now
        raise NotImplementedError(f"Input should be square. Please double check the input.")
    return [cv2.resize(pcd, (input_size, input_size), interpolation=cv2.INTER_NEAREST) for pcd in pointclouds]


def extract_fpfh_feature(
    pcd: np.ndarray,
    radius: float = 0.03,
    max_nn_normal: int = 300,
    max_nn_feature: int = 600,
) -> torch.tensor:
    """
    Compute FPFH feature of an input point cloud
    Args:
        pcd:         numpy array. H W 3. point cloud
        radius       radius to compute geometric features.

    Returns:
        fpfh feature: torch.tensor H W 33
    """
    ori_h, ori_w, _ = pcd.shape
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.reshape(-1, 3)))
    radius_normal = radius
    radius_feature = 2.5 * radius
    o3d_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn_normal))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        o3d_pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn_feature),
    )
    return torch.tensor(pcd_fpfh.data.copy().reshape(33, ori_h, ori_w)).permute(1, 2, 0).float()


def get_alignz_campose(cam_pose):
    new_x_xyplane = cam_pose[:3, 2].copy()
    new_x_xyplane[2] = 0
    new_x_xyplane = new_x_xyplane / (np.linalg.norm(new_x_xyplane) + 1e-5)
    new_y_xyplane = np.array([-new_x_xyplane[1], new_x_xyplane[0], 0])
    new_z = np.cross(new_x_xyplane, new_y_xyplane)
    new_z = new_z / np.linalg.norm(new_z)
    updated_campose = np.eye(4)
    updated_campose[:3, :3] = np.array([new_x_xyplane, new_y_xyplane, new_z]).T
    return updated_campose


def load_pickle_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
