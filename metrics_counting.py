# -*- coding: utf-8 -*-
import os
import sys
import warnings
from argparse import ArgumentParser
import glob
import cv2
import numpy as np
import torch
from PIL import Image
from statistics import mean

from config import (DEVICE, INFERENCE_SIZE, IOU_THRESHOLD, KPTS_CONF,
                    MAX_OBJECT_CNT, PERSON_CONF, XMEM_CONFIG)
from inference.inference_utils import (add_new_classes_to_dict,
                                       generate_colors_dict,
                                       get_iou_filtered_yolo_mask_bboxes,
                                       merge_masks, overlay_mask_on_image)
from inference.interact.interactive_utils import torch_prob_to_numpy_mask
from metrics.metrics_utils import (calculate_tp_fp_fn, metrics_calculation,
                                   update_gt_pred_mapping_dict,
                                   yolo_bboxes_from_txt)
from tracker import Tracker
from pose_estimation import Yolov8PoseModel


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = ArgumentParser()
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Directory with Yolo labels (txt-files)')
    parser.add_argument(
        '--width', type=int, default=INFERENCE_SIZE[0], required=False, help='Inference width')
    parser.add_argument(
        '--height', type=int, default=INFERENCE_SIZE[1], required=False, help='Inference height')
    parser.add_argument('--device', type=str, default=DEVICE,
                        required=False, help='GPU id')
    parser.add_argument('--person_conf', type=float, default=PERSON_CONF,
                        required=False, help='YOLO person confidence')
    parser.add_argument('--kpts_conf', type=float, default=KPTS_CONF,
                        required=False, help='YOLO keypoints confidence')
    parser.add_argument('--iou_thresh', type=float, default=IOU_THRESHOLD,
                        required=False, help='IOU threshold to find new persons bboxes')
    parser.add_argument('--print_every', type=int, default=10,
                        required=False, help='Print metrics every N frames')
    args = parser.parse_args()

    if torch.cuda.device_count() > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)

    FINAL_PRECISION = 0.
    FINAL_RECALL = 0.
    FINAL_F1_SCORE = 0.

    precision_arr = []
    recall_arr = []
    f1_score_arr = []

    persons_in_video = False

    class_color_mapping = generate_colors_dict(MAX_OBJECT_CNT+1)

    current_frame_index = 0
    precision_sum = 0.
    recall_sum = 0.
    f1_score_sum = 0.
    gt_pred_mapping_dict = {}
    class_label_mapping = {}

    for video_dir in os.listdir(args.labels_dir):
        yolo_files = sorted(glob.glob(f'{args.labels_dir}/{video_dir}/obj_train_data/*.txt'))
        image_files = sorted([file for file in os.listdir(f'{args.labels_dir}/{video_dir}/obj_train_data') if
                              file.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))])


        yolov8pose_model = Yolov8PoseModel(DEVICE, PERSON_CONF, KPTS_CONF)
        tracker = Tracker(XMEM_CONFIG, MAX_OBJECT_CNT, DEVICE)
        persons_in_video = False

        class_color_mapping = generate_colors_dict(MAX_OBJECT_CNT + 1)

        current_frame_index = -1

        mapping_dict = {}
        class_label_mapping = {}

        with torch.cuda.amp.autocast(enabled=True):

            for i, frame_yolo in enumerate(yolo_files):
                current_frame_index += 1

                frame = cv2.imread(os.path.join(f'{args.labels_dir}/{video_dir}/obj_train_data', image_files[i]))

                frame = cv2.resize(frame, INFERENCE_SIZE, interpolation=cv2.INTER_AREA)

                gt_bboxes_with_idx = yolo_bboxes_from_txt(frame_yolo, INFERENCE_SIZE[0], INFERENCE_SIZE[1])
                gt_bboxes_without_idx = torch.tensor([box[1:] for box in gt_bboxes_with_idx])

                yolo_filtered_bboxes = yolov8pose_model.get_filtered_bboxes_by_confidence(frame)

                if len(yolo_filtered_bboxes) > 0:
                    persons_in_video = True
                else:
                    masks = []
                    continue

                if persons_in_video:
                    if len(class_label_mapping) == 0:  # First persons in video
                        mask = tracker.create_mask_from_img(frame, yolo_filtered_bboxes, device='0')
                        unique_labels = np.unique(mask)
                        class_label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
                        mask = np.array([class_label_mapping[label] for label in mask.flat]).reshape(mask.shape)
                        prediction = tracker.add_mask(frame, mask)
                    elif len(filtered_bboxes) > 0:  # Additional/new persons in video
                        mask = tracker.create_mask_from_img(frame, filtered_bboxes, device='0')
                        unique_labels = np.unique(mask)
                        mask_image = Image.fromarray(mask, mode='L')
                        class_label_mapping = add_new_classes_to_dict(unique_labels, class_label_mapping)
                        mask = np.array([class_label_mapping[label] for label in mask.flat]).reshape(mask.shape)
                        merged_mask = merge_masks(masks.squeeze(0), torch.tensor(mask))
                        prediction = tracker.add_mask(frame, merged_mask.squeeze(0).numpy())
                        filtered_bboxes = []
                    else:  # Only predict
                        prediction = tracker.predict(frame)

                    masks = torch.tensor(torch_prob_to_numpy_mask(prediction)).unsqueeze(0)
                    masks = tracker.keep_largest_connected_components(masks)
                    mask_bboxes_with_idx = tracker.masks_to_boxes_with_ids(masks)

                    # dict which mathes gt_classes and model_classes
                    mapping_dict = update_gt_pred_mapping_dict(mapping_dict, gt_bboxes_with_idx, mask_bboxes_with_idx)

                    filtered_bboxes = get_iou_filtered_yolo_mask_bboxes(yolo_filtered_bboxes, mask_bboxes_with_idx, 0.1)


                    # Metrics calculation
                tp, fp, fn = calculate_tp_fp_fn(gt_bboxes_with_idx, mask_bboxes_with_idx, mapping_dict,
                                                iou_threshold=0.5)
                precision, recall, f1_score = metrics_calculation(tp, fp, fn)

                precision_arr.append(precision)
                recall_arr.append(recall)
                f1_score_arr.append(f1_score)

                if current_frame_index % args.print_every == 0:
                    print(f'current precision on {video_dir}: {precision}')
                    print(f'current recall on {video_dir}: {recall}')
                    print(f'current f1_score on {video_dir}: {f1_score}')
                    print('-' * 8)
                    print(f'overall precision: {mean(precision_arr)}')
                    print(f'overall recall: {mean(recall_arr)}')
                    print(f'overall f1_score: {mean(f1_score_arr)}')

    print('=' * 8)
    print(f'FINAL PRECICION: {mean(precision_arr)}')
    print(f'FINAL RECALL: {mean(recall_arr)}')
    print(f'FINAL F1 SCORE: {mean(f1_score_arr)}')
