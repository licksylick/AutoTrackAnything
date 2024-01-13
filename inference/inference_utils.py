# -*- coding: utf-8 -*-
import os
import random

import cv2
import numpy as np
import torch
from mobile_sam import (SamAutomaticMaskGenerator, SamPredictor,
                        sam_model_registry)
from PIL import Image
from skimage.measure import label, regionprops

from config import (DEVICE, IOU_THRESHOLD, KPTS_CONF, MAX_OBJECT_CNT,
                    PERSON_CONF, XMEM_CONFIG)

os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE

torch.cuda.empty_cache()


def generate_colors_dict(num_classes):
    if num_classes > 256*256*256:
        raise ValueError('Number of classes is too large.')

    color_dict = {}

    for class_id in range(num_classes):
        while True:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)

            is_maximally_different = True
            for existing_color in color_dict.values():
                distance = ((r - existing_color[0]) ** 2 +
                            (g - existing_color[1]) ** 2 +
                            (b - existing_color[2]) ** 2) ** 0.5

                if distance < 100:
                    is_maximally_different = False
                    break

            if is_maximally_different:
                color_dict[class_id] = (r, g, b)
                break
    return color_dict


def masks_on_im(masks, image):
    # Make sure the dtype matches image
    result = np.zeros_like(image, dtype=np.uint8)

    for mask in masks:
        color = np.random.randint(0, 256, size=3)

        colored_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        masked_region = colored_mask * color
        result += masked_region.astype(np.uint8)

    return result


def create_mask_from_img(image, yolov7_bboxes, save_path=None, sam_checkpoint='./saves/mobile_sam.pt', model_type='vit_t', device='0'):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=f'cuda:{device}')

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    input_boxes = torch.tensor(yolov7_bboxes, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    for i, mask in enumerate(masks):

        binary_mask = masks[i].cpu().squeeze().numpy().astype(np.uint8)

        contours, hierarchy = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key=cv2.contourArea)

        bbox = [int(x) for x in cv2.boundingRect(largest_contour)]

        segmentation = largest_contour.flatten().tolist()
        mask = segmentation

        height, width, _ = image.shape

        mask = np.array(mask).reshape(-1, 2)

        mask_norm = mask / np.array([width, height])

        xmin, ymin = mask_norm.min(axis=0)
        xmax, ymax = mask_norm.max(axis=0)
        bbox_norm = np.array([xmin, ymin, xmax, ymax])

        yolo = np.concatenate([bbox_norm, mask_norm.reshape(-1)])

    result = masks_on_im(
        [mask.cpu().squeeze().numpy().astype(np.uint8) for mask in masks], image)

    result = result[:, :, 0]
    if save_path is not None:
        # 'L' mode indicates grayscale
        mask_image = Image.fromarray(result, mode='L')
        mask_image.save(save_path)

    return result


def masks_to_boxes_with_ids(mask_tensor: torch.Tensor) -> torch.Tensor:
    unique_values = torch.unique(mask_tensor[mask_tensor != 0])

    bbox_list = []

    for unique_value in unique_values:
        binary_mask = (mask_tensor == unique_value).byte()

        nonzero_coords = torch.nonzero(binary_mask, as_tuple=False)

        if nonzero_coords.numel() > 0:
            min_x = torch.min(nonzero_coords[:, 2])
            min_y = torch.min(nonzero_coords[:, 1])
            max_x = torch.max(nonzero_coords[:, 2])
            max_y = torch.max(nonzero_coords[:, 1])

            bbox = [unique_value.item(), min_x.item(), min_y.item(),
                    max_x.item(), max_y.item()]
            bbox_list.append(bbox)

    return bbox_list


def add_new_classes_to_dict(unique_labels, class_label_mapping):
    for pixel_value in unique_labels:
        last_id = list(class_label_mapping.items())[-1][1]
        new_id = last_id + 1
        class_label_mapping[pixel_value] = new_id
        class_label_mapping[0] = 0
    return class_label_mapping


def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[1:5]

    x1_intersection = max(x1_1, x1_2)
    y1_intersection = max(y1_1, y1_2)
    x2_intersection = min(x2_1, x2_2)
    y2_intersection = min(y2_1, y2_2)

    intersection_area = max(0, x2_intersection - x1_intersection + 1) * \
        max(0, y2_intersection - y1_intersection + 1)

    area_box1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    area_box2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

    iou = intersection_area / float(area_box1 + area_box2 - intersection_area)

    return iou


def get_iou_filtered_yolo_mask_bboxes(boxes1, boxes2, iou_threshold=0.2):
    filtered_boxes = []

    for box1 in boxes1:
        iou_scores = [calculate_iou(box1, box2) for box2 in boxes2]
        if all(iou < iou_threshold for iou in iou_scores):
            filtered_boxes.append(box1)

    return filtered_boxes


def merge_masks(mask1, mask2):
    unique_mask1 = torch.unique(mask1)
    unique_mask2 = torch.unique(mask2)

    merged_classes = torch.unique(torch.cat((unique_mask1, unique_mask2)))
    merged_mask = torch.zeros_like(mask2)

    for idx, cls in enumerate(merged_classes):
        merged_mask[(mask1 == cls) | (mask2 == cls)] = idx

    return merged_mask


def save_mask(mask_tensor, colors_dict, current_frame_index):
    if len(colors_dict) > 1:
        unique_labels = colors_dict.keys()
    else:
        unique_labels = []

    color_mapping = {value: key for (key, value) in colors_dict.items()}
    mask_numpy = mask_tensor.squeeze().cpu().numpy()

    height, width = mask_numpy.shape
    grayscale_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            class_label = mask_numpy[i, j]
            if not class_label in color_mapping.keys():
                grayscale_image[i, j] = 0
            else:
                grayscale_image[i, j] = color_mapping[class_label]

    mask_image = Image.fromarray(grayscale_image, mode='L')

    mask_image.save(f'masks/mask_on_frame_{current_frame_index}.png')


def overlay_mask_on_image(image, mask, class_color_mapping, alpha=0.5):
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
    result_image = image.copy()

    for class_label, color in class_color_mapping.items():
        if class_label == 0:  # Skip background class
            continue
        class_mask = (mask_np == class_label).astype(np.uint8)
        result_image[class_mask == 1] = (
            1 - alpha) * result_image[class_mask == 1] + alpha * np.array(color)

    return result_image


def filter_yolov7_bboxes_by_size(yolov7bboxes, image_size, percentage=10):
    min_bbox_width = image_size[0] * (percentage/100)
    min_bbox_height = image_size[1] * (percentage/100)

    filtered_bboxes = []

    for bbox in yolov7bboxes:
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        if bbox_width >= min_bbox_width and bbox_height >= min_bbox_height:
            filtered_bboxes.append(bbox)

    return filtered_bboxes
