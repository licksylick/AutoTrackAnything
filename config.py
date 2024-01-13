# -*- coding: utf-8 -*-

DEVICE = '0' # For GPU set device num which you want to use (or set 'cpu', but it's too slow)
#DEVICE = 'cpu'

# Our confidence for every person (bbox)
PERSON_CONF = 0.7

KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

# Our confidence for used keypoints
KPTS_CONF = 0.6

IOU_THRESHOLD = 0.1

# It's xMem original config, you can try to change this values for your task (check xMem article)
XMEM_CONFIG = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 256,
    'min_mid_term_frames': 7,
    'max_mid_term_frames': 20,
    'max_long_term_elements': 10000,
}

# Max possible count of persons in video (if you has error, set bigger number)
MAX_OBJECT_CNT = 10

# Check new persons in frame every N frames
YOLO_EVERY = 20

# Resize processed video. For better results you can increase resolution
INFERENCE_SIZE = (960, 500)
