import base64
import os

import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from segment_anything import SamPredictor, sam_model_registry

import kalm.configs.local_config as local_config
from kalm.utils import (
    approx_equal,
    draw_grid,
    draw_seg_on_im,
    extract_tag_content,
    is_degenerated_mask,
    resize_to,
)

GRID_SHAPE = (5, 5)

TASKNAME2DESC = {
    "drawer": "Opening the top drawer.",
    "coffee": "Lifting the handle of the coffee machine.",
    "pour": "Pouring something into the bowl.",
}


class GPTClient:
    def __init__(self, cache_dir="tmp_gptclient"):
        self.client = OpenAI(
            organization=local_config.GPT_ORGANIZATION_KEY,
            project=local_config.GPT_PROJECT_KEY,
            api_key=local_config.GPT_API_KEY,
        )

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        self.default_tmp_save_name = os.path.join(cache_dir, "tmp_save.png")

    def encode_image(self, image_path):
        """
        encode image at image_path to base64 string to send to GPT
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def obtain_GPT_coordinate(
        self,
        task_desc,
        sequence: np.ndarray,
        grid_shape: tuple[int],
        outerloop_iter_k=0,
    ):
        """
        Prompt GPT to obtain an x,y coordinate of the task relevant part.
        Args:
            task_desc:
            sequence: rgb image sequence
            vis:

        Returns:

        """

        instruction = f'We wish complete the task of "{task_desc}". The provided images are sampled from the demonstration of the correct trajectory to complete this task.'
        img_prompt = "Please identify the object parts that is relevant to the agent's action."
        img_prompt += "For example, if the agent is trying to grasp something, please identify the object part in the scene that the held object is interacted with."
        img_prompt += "For example, if the agent is trying to put something into a container, please identify the object part of the container that the agent is interacting with."
        img_prompt += "Let's think step by step folling the following pattern:" "task = '...' # Describe the task" "object_parts_in_the_task = ['object_1'] # Identify the object in the task"
        img_prompt += "After analyzing the object part, please localize the object part in the given image." "We have provided a coordinate grid to help you identify the target object part. " "Please return the grid IDs of the object part in the image. " "<output>{grid_id1} {grid_id2} ...</output>"

        images = list()
        b64_images = list()  #  base64 encoded images
        for i, img in enumerate(sequence):
            img = draw_grid(
                img,
                nr_vertical=grid_shape[1],
                nr_horizontal=grid_shape[0],
                resize_to_max_dim=512,
            )
            images.append(img)
            save_path = os.path.join(
                self.cache_dir,
                f"iter_{outerloop_iter_k:02d}_ref_imageswithgrids_{i}.png",
            )
            cv2.imwrite(save_path, img[..., ::-1])
            encoded_file = cv2.imencode(".jpg", img[..., ::-1])
            encoded_image = base64.b64encode(encoded_file[1]).decode("utf-8")
            b64_images.append(encoded_image)

        messages = [
            {
                "role": "user",
                "content": [
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{im_i}"},
                        }
                        for im_i in b64_images
                    ],
                    {"type": "text", "text": instruction + "\n" + img_prompt},
                ],
            }
        ]

        gpt_return_result = self.client.chat.completions.create(model="gpt-4o", messages=messages)
        return messages, gpt_return_result

    def obtain_GPT_coordinate_auto(
        self,
        task_desc,
        sequence: np.ndarray,
        grid_shape: tuple[int, int],
        max_retries: int = 3,
    ) -> tuple[list, list]:
        for i in range(max_retries):
            try:
                messages, result = self.obtain_GPT_coordinate(task_desc, sequence, grid_shape)
                x_firstframe = extract_tag_content(result.choices[0].message.content, "output")
                all_grids_firstframe = list(map(int, x_firstframe.split(" ")))

                # Return both the messages and the result contactenated. This is useful for future prompting.
                messages = messages.copy()
                messages.append({"role": "user", "content": result.choices[0].message.content})

                return all_grids_firstframe, messages
            except Exception as e:
                print(f"Error in GPT call. Retrying. Error: {e}")
        raise RuntimeError("GPT call failed.")

    def obtain_GPT_mask(
        self,
        previous_messages,
        image_with_masks_firstframe,
        task_desc,
        gpt_prev_iter_msg=None,
        iter_n=0,
    ):
        b64_images = list()  #  base64 encoded images
        if isinstance(image_with_masks_firstframe, list):
            instruction = "I have provided you a sequence of images. They are all based on the same image but highlighted with different masks.\n"
            instruction += f"Please select the ID of the mask that has been previously identified as the most relevant part " f"of the object that is moved by the actuated agent, for task '{task_desc}'.\n"
            instruction += "Put your answer in the format of <output>{ID}</output>\n"
            instruction += "Let's think step by step following the pattern:"
            instruction += "image_1_mask = '...' # Describe the mask in the first image"
            instruction += "image_2_mask = '...' # Describe the mask in the second image"
            instruction += "..."
            instruction += "matched_mask_ids = ['ID1', 'ID2', ...] # List all the IDs of the matched masks in the sequence"
            instruction += "output: <output>{ID}</output>  # return the first frame mask ID that matches the mask description."

            for i, img in enumerate(image_with_masks_firstframe):
                img = resize_to(img, 512)
                save_path = os.path.join(self.cache_dir, f"iter_{iter_n:02d}_sammasks_framefirst_{i}.png")
                cv2.imwrite(save_path, img[..., ::-1])
                encoded_file = cv2.imencode(".jpg", img[..., ::-1])
                encoded_image = base64.b64encode(encoded_file[1]).decode("utf-8")
                b64_images.append(encoded_image)
        else:
            raise NotImplementedError("Not updated.")

        if gpt_prev_iter_msg is not None:
            instruction += f"The mask you selected previously is shown in red. That is not the correct option. Please select another one."
            img = resize_to(gpt_prev_iter_msg, 512)
            save_path = os.path.join(self.cache_dir, f"sammasks_previter_{iter_n}.png")
            cv2.imwrite(save_path, img[..., ::-1])
            encoded_file = cv2.imencode(".jpg", img[..., ::-1])
            encoded_image = base64.b64encode(encoded_file[1]).decode("utf-8")
            b64_images.append(encoded_image)

        messages = previous_messages + [
            {
                "role": "user",
                "content": [
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{im_i}"},
                        }
                        for im_i in b64_images
                    ],
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        gpt_return_result = self.client.chat.completions.create(model="gpt-4o", messages=messages)
        return gpt_return_result, messages

    def obtain_GPT_mask_auto(
        self,
        previous_messages,
        image_with_masks_firstframe,
        task_desc,
        max_retries=3,
        gpt_prev_iter_msg=None,
        iter_n=0,
    ):
        for i in range(max_retries):
            try:
                result, messages = self.obtain_GPT_mask(
                    previous_messages,
                    image_with_masks_firstframe,
                    task_desc,
                    gpt_prev_iter_msg,
                    iter_n=iter_n,
                )
                x_firstframe = extract_tag_content(result.choices[0].message.content, "output")
                x_firstframe = int(x_firstframe)

                # Return both the messages and the result contactenated. This is useful for future prompting.
                messages = previous_messages.copy()
                messages.append({"role": "user", "content": result.choices[0].message.content})

                return x_firstframe, messages
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Error in GPT call. Retrying. Error: {e}")
        raise RuntimeError("GPT call failed.")

    def obtain_GPT_mask_on_query_image(
        self,
        previous_messages,
        ref_image_with_masks,
        sequence,
        query_image,
        task_desc,
        grid_shape: tuple[int],
        query_im_index=0,
        iter_n=0,
    ):
        """
        Prompt GPT to obtain an x,y coordinate of the task relevant part.
        Args:
            task_desc:
            sequence:
            query_image: numpy 0-255
            vis:

        Returns:

        """
        instruction = f'We wish complete the task of "{task_desc}". ' f"We have provided image sequence sampled from the demonstration of the correct trajectory to complete this task."
        img_prompt = "We wish to complete the same task in another scene as shown in the last image."
        img_prompt += "We have provided the mask you previously selected in the image sequence in the first image."
        img_prompt += f"Please identify the object part that the actuated agent need to interact with in the last provided image for completing the task of '{task_desc}'."
        img_prompt += "Each image contains a coordinate grid to help you identify the target object part. "
        ending_prompt = "Return the coordinate of this object part in the final image, utilizing the labelled coordinate grid for guidance. " "Please return IDs of all the grid cells that contains the object part, in the format of <output>{ID1} {ID2} ... {ID_N}</output>." "Let's think step by step. First describe the task, then identify the object part that" "is similar to that in the first image, and finally return the grid IDs of the object."

        b64_images = list()  #  base64 encoded images

        image_with_masks = resize_to(ref_image_with_masks, 512)
        encoded_file = cv2.imencode(".jpg", image_with_masks[..., ::-1])
        encoded_image = base64.b64encode(encoded_file[1]).decode("utf-8")
        # Insert to second to last.
        b64_images.append(encoded_image)

        for img in sequence:
            img = draw_grid(
                img,
                nr_vertical=grid_shape[1],
                nr_horizontal=grid_shape[0],
                resize_to_max_dim=512,
            )
            encoded_file = cv2.imencode(".jpg", img[..., ::-1])
            encoded_image = base64.b64encode(encoded_file[1]).decode("utf-8")
            b64_images.append(encoded_image)

        query_image = draw_grid(
            query_image,
            nr_vertical=grid_shape[1],
            nr_horizontal=grid_shape[0],
            resize_to_max_dim=512,
        )
        save_path = os.path.join(self.cache_dir, f"iter_{iter_n:02d}_queryimagegrids_{query_im_index}.png")
        cv2.imwrite(save_path, query_image[..., ::-1])
        encoded_file = cv2.imencode(".jpg", query_image[..., ::-1])
        encoded_image = base64.b64encode(encoded_file[1]).decode("utf-8")
        b64_images.append(encoded_image)

        messages = previous_messages + [
            {
                "role": "user",
                "content": [
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{im_i}"},
                        }
                        for im_i in b64_images
                    ],
                    {
                        "type": "text",
                        "text": instruction + "\n" + img_prompt + "\n" + ending_prompt,
                    },
                ],
            }
        ]

        gpt_return_result = self.client.chat.completions.create(model="gpt-4o", messages=messages)
        return gpt_return_result, messages

    def obtain_GPT_mask_on_query_image_auto(self, gpt_client_info, max_retries=3, query_im_index=0, iter_n=0):
        for i in range(max_retries):
            try:
                result, messages = self.obtain_GPT_mask_on_query_image(
                    gpt_client_info.previous_messages,
                    gpt_client_info.ref_image_with_masks,
                    gpt_client_info.sequence,
                    gpt_client_info.query_image,
                    gpt_client_info.task_desc,
                    gpt_client_info.grid_shape,
                    query_im_index=query_im_index,
                    iter_n=iter_n
                )
                x = extract_tag_content(result.choices[0].message.content, "output")
                all_grids = list(map(int, x.split(" ")))

                # Return both the messages and the result contactenated. This is useful for future prompting.
                messages = gpt_client_info.previous_messages.copy()
                messages.append({"role": "user", "content": result.choices[0].message.content})

                return all_grids, messages
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Error in GPT call. Retrying. Error: {e}")
        raise RuntimeError("GPT call failed.")


class MaskPredictor:
    def __init__(self, sam_ckpt_path=None, device=None):
        if sam_ckpt_path is None:
            sam_ckpt_path = local_config.SAM_CKPT

        if sam_ckpt_path is None:
            cache_dir = os.path.expanduser('./cache')
            sam_ckpt_path = os.path.join(cache_dir, "sam_vit_h_4b8939.pth")
            if not os.path.exists(sam_ckpt_path):
                os.makedirs(cache_dir, exist_ok=True)
                print(f'Downloading SAM checkpoint to {sam_ckpt_path}')
                torch.hub.download_url_to_file('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth', sam_ckpt_path)

        self.sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to(device)
        self.predictor = SamPredictor(self.sam)

    def get_candidate_masks_in_grid(
        self,
        im: np.ndarray,
        top: float,
        left: float,
        bottom: float,
        right: float,
        min_distance: float = 10,
        conf_thr=0.88,
        pcd: np.ndarray = None,
        remove_degenerate: bool = True,
        vis: bool = False,
    ) -> tuple:
        self.predictor.set_image(im)

        points_horizontal = np.linspace(left, right, int((right - left) / min_distance) + 1)
        points_vertical = np.linspace(top, bottom, int((bottom - top) / min_distance) + 1)

        if vis:
            # Plot all the points
            plt.imshow(im)
            for x in points_horizontal:
                for y in points_vertical:
                    plt.scatter(x, y, c="r")
            plt.show()
            plt.close()

        masks_all = []
        scores_all = []
        points_all = []

        for x in points_horizontal:
            for y in points_vertical:
                input_point_xy = np.array([[x, y]])
                input_label = np.array([1])
                masks, scores, _ = self.predictor.predict(
                    point_coords=input_point_xy,
                    point_labels=input_label,
                    multimask_output=True,  # Set to True for better score.
                )
                mask_list = []
                for mask, score in zip(masks, scores):
                    mask_list.append([mask, score])
                mask_list.sort(key=lambda x: x[1], reverse=True)
                mask, score = mask_list[0]
                for mask, score in mask_list:
                    add_to_mask_lib = False
                    if score >= conf_thr:
                        if remove_degenerate:
                            cleaned_mask, is_degenerate = is_degenerated_mask(mask, pcd)
                            if is_degenerate:
                                continue
                        add_to_mask_lib = True
                        for ix, imask in enumerate(masks_all):
                            if approx_equal(mask, imask):
                                add_to_mask_lib = False
                                if score > scores_all[ix]:
                                    masks_all[ix] = mask
                                break
                    if add_to_mask_lib:
                        points_all.append(input_point_xy)
                        masks_all.append(mask)
                        scores_all.append(score)

                        if vis:
                            # Plot the mask and the point on the image
                            plt.imshow(draw_seg_on_im(im, [mask], alpha=0.6))
                            plt.scatter(x, y, c="r")
                            plt.title(f"Score: {score}")
                            plt.show()
                            plt.close()
        return masks_all, scores_all, points_all
