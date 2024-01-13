def yolo_bboxes_from_txt(file_path, image_width=1920., image_height=1080.):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    for line in lines:
        parts = line.strip().split()
        idx = int(parts[0])
        x_rel = float(parts[1])
        y_rel = float(parts[2])
        width_rel = float(parts[3])
        height_rel = float(parts[4])

        x1 = (x_rel - width_rel / 2) * image_width
        y1 = (y_rel - height_rel / 2) * image_height
        x2 = (x_rel + width_rel / 2) * image_width
        y2 = (y_rel + height_rel / 2) * image_height

        bbox = [idx, x1, y1, x2, y2]
        bboxes.append(bbox)
    
    sorted_bboxes = sorted(bboxes, key=lambda bbox: bbox[0])
    return sorted_bboxes

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection / (area_box1 + area_box2 - intersection)
    return iou


def update_gt_pred_mapping_dict(mapping_dict, gt_bboxes_with_idx, mask_bboxes_with_idx):
    for id_box in mask_bboxes_with_idx:
        max_iou = 0
        matching_class = None

        for gt_box in gt_bboxes_with_idx:
            iou = calculate_iou(id_box[1:], gt_box[1:])
            if iou > max_iou:
                max_iou = iou
                matching_class = int(gt_box[0])

        if matching_class is not None:
            mapping_dict[int(id_box[0])] = matching_class
    return mapping_dict


def calculate_tp_fp_fn(gt_bboxes_with_idx, mask_bboxes_with_idx, mapping_dict, iou_threshold):
    tp = 0
    fp = 0
    fn = 0

    for id_box in mask_bboxes_with_idx:
        max_iou = 0
        matching_gt_box = None
        for gt_box in gt_bboxes_with_idx:
            iou = calculate_iou(id_box[1:], gt_box[1:])
            if iou > max_iou and mapping_dict[int(id_box[0])] == int(gt_box[0]):
                max_iou = iou
                matching_gt_box = gt_box
        if max_iou >= iou_threshold:
            tp += 1
        else:
            fp += 1

    fn = len(gt_bboxes_with_idx) - tp
    return tp, fp, fn


def metrics_calculation(tp, fp, fn):
    if tp == 0:
        precision = 0
        recall = 0
        f1_score = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1_score