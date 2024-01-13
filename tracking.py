# -*- coding: utf-8 -*-
import csv
import os
import sys
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import (DEVICE, INFERENCE_SIZE, IOU_THRESHOLD, KPTS_CONF,
                    MAX_OBJECT_CNT, PERSON_CONF, XMEM_CONFIG, YOLO_EVERY)
from inference.inference_utils import (add_new_classes_to_dict,
                                       generate_colors_dict,
                                       get_iou_filtered_yolo_mask_bboxes,
                                       merge_masks, overlay_mask_on_image)
from inference.interact.interactive_utils import torch_prob_to_numpy_mask
from tracker import Tracker
from pose_estimation import Yolov8PoseModel

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = ArgumentParser()
    parser.add_argument('--video_path', type=str,
                        required=True, help='Path to input video')
    parser.add_argument(
        '--width', type=int, default=INFERENCE_SIZE[0], required=False, help='Inference width')
    parser.add_argument(
        '--height', type=int, default=INFERENCE_SIZE[1], required=False, help='Inference height')
    parser.add_argument('--frames_to_propagate', type=int,
                        default=None, required=False, help='Frames to propagate')
    parser.add_argument('--output_video_path', type=str, default=None,
                        required=False, help='Output video path to save')
    parser.add_argument('--device', type=str, default=DEVICE,
                        required=False, help='GPU id')
    parser.add_argument('--person_conf', type=float, default=PERSON_CONF,
                        required=False, help='YOLO person confidence')
    parser.add_argument('--kpts_conf', type=float, default=KPTS_CONF,
                        required=False, help='YOLO keypoints confidence')
    parser.add_argument('--iou_thresh', type=float, default=IOU_THRESHOLD,
                        required=False, help='IOU threshold to find new persons bboxes')
    parser.add_argument('--yolo_every', type=int, default=YOLO_EVERY,
                        required=False, help='Find new persons with YOLO every N frames')
    parser.add_argument('--output_path', type=str,
                        default='tracking_results.csv', required=False, help='Output filepath')

    args = parser.parse_args()

    if torch.cuda.device_count() > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)

    cap = cv2.VideoCapture(args.video_path)
    df = pd.DataFrame(
        columns=['frame_id', 'person_id', 'x1', 'y1', 'x2', 'y2'])

    if args.output_video_path is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        result = cv2.VideoWriter(args.output_video_path, cv2.VideoWriter_fourcc(
            'm', 'p', '4', 'v'), fps, (args.width, args.height))

    yolov8pose_model = Yolov8PoseModel(DEVICE, PERSON_CONF, KPTS_CONF)
    tracker = Tracker(XMEM_CONFIG, MAX_OBJECT_CNT, DEVICE)
    persons_in_video = False

    class_color_mapping = generate_colors_dict(MAX_OBJECT_CNT+1)

    current_frame_index = 0
    class_label_mapping = {}

    with torch.cuda.amp.autocast(enabled=True):

        while (cap.isOpened()):
            _, frame = cap.read()

            if frame is None or (args.frames_to_propagate is not None and current_frame_index == args.frames_to_propagate):
                break

            frame = cv2.resize(frame, (args.width, args.height),
                               interpolation=cv2.INTER_AREA)
            if current_frame_index % args.yolo_every == 0:
                yolo_filtered_bboxes = yolov8pose_model.get_filtered_bboxes_by_confidence(frame)

            if len(yolo_filtered_bboxes) > 0:
                persons_in_video = True
            else:
                masks = []
                mask_bboxes_with_idx = []

            if persons_in_video:
                if len(class_label_mapping) == 0:  # First persons in video
                    mask = tracker.create_mask_from_img(
                        frame, yolo_filtered_bboxes, device='0')
                    unique_labels = np.unique(mask)
                    class_label_mapping = {
                        label: idx for idx, label in enumerate(unique_labels)}
                    mask = np.array([class_label_mapping[label]
                                    for label in mask.flat]).reshape(mask.shape)
                    prediction = tracker.add_mask(frame, mask)
                elif len(filtered_bboxes) > 0:  # Additional/new persons in video
                    mask = tracker.create_mask_from_img(
                        frame, filtered_bboxes, device='0')
                    unique_labels = np.unique(mask)
                    mask_image = Image.fromarray(mask, mode='L')
                    class_label_mapping = add_new_classes_to_dict(
                        unique_labels, class_label_mapping)
                    mask = np.array([class_label_mapping[label]
                                    for label in mask.flat]).reshape(mask.shape)
                    merged_mask = merge_masks(
                        masks.squeeze(0), torch.tensor(mask))
                    prediction = tracker.add_mask(
                        frame, merged_mask.squeeze(0).numpy())
                    filtered_bboxes = []
                else:  # Only predict
                    prediction = tracker.predict(frame)

                masks = torch.tensor(
                    torch_prob_to_numpy_mask(prediction)).unsqueeze(0)
                mask_bboxes_with_idx = tracker.masks_to_boxes_with_ids(masks)

                if current_frame_index % args.yolo_every == 0:
                    filtered_bboxes = get_iou_filtered_yolo_mask_bboxes(
                        yolo_filtered_bboxes, mask_bboxes_with_idx, iou_threshold=args.iou_thresh)

                    # VISUALIZATION
            if args.output_video_path is not None:
                if len(mask_bboxes_with_idx) > 0:
                    for bbox in mask_bboxes_with_idx:
                        cv2.rectangle(frame, (int(bbox[1]), int(bbox[2])), (int(
                            bbox[3]), int(bbox[4])), (255, 255, 0), 2)
                        cv2.putText(frame, f'{bbox[0]}', (int(
                            bbox[1])-10, int(bbox[2])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    visualization = overlay_mask_on_image(
                        frame, masks, class_color_mapping, alpha=0.75)
                    visualization = cv2.cvtColor(
                        visualization, cv2.COLOR_BGR2RGB)
                    result.write(visualization)
                else:
                    result.write(frame)

            if len(mask_bboxes_with_idx) > 0:
                for bbox in mask_bboxes_with_idx:
                    person_id = bbox[0]
                    x1 = bbox[1]
                    y1 = bbox[2]
                    x2 = bbox[3]
                    y2 = bbox[4]
                    df.loc[len(df.index)] = [
                        int(current_frame_index), person_id, x1, y1, x2, y2]
            else:
                df.loc[len(df.index)] = [int(current_frame_index),
                                         None, None, None, None, None]
            print(
                f'current_frame_index: {current_frame_index}, persons in frame: {len(mask_bboxes_with_idx)}')
            current_frame_index += 1

    df.to_csv(args.output_path, index=False)
    if args.output_video_path is not None:
        result.release()
