# AutoTrackAnything

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NLLtHFcoPH-vncLH_pAwGI4wpH5PhGol?usp=sharing)

<div style="display: flex; align-items: center; justify-content: space-around;">
  <img src="media/RES_OUT_OF_FRAME.gif" height="200">
  <img src="media/FILTERED_OCCLUSIONS.gif" height="200">
</div>
  
-----
## 🔥 Advantages  
* Automatic creation of object masks for further tracking.
* New objects are added automatically.
* Tracking even if the object is out of frame.
* Tracking even if there are a large number of occlusions (intersections) of objects.
* Easy to use.
* Easy to change for any task.

-----

## ⚠️ Some necessary information
It's multipurpose tracking approach using Yolov8, SAM, xMem and my wrapper and algorithms.  
In this case it's uses for person detection, but you can simply change task (see [point 4](https://github.com/licksylick/AutoTrackAnything?tab=readme-ov-file#-4-use-project-for-your-custom-tasks)).  
And I use keypoints confidence for adding good visible persons (you can remove it later).
   
  
It's not a super-approach, so maybe you will need to set hyperparameters or train models for your task. But it's very useful and easy to start project, that you can use for multiple object tracking.  
On my task (person tracking) it works better that other approaches: MOT, ByteTrack, DeepSort, Kalman FIlter etc.

-----

## ✅ 1. Preparing
### Install all necessary libs:
  ```sh
  pip3 install -r requirements.txt
  ```
Note: if you are using a GPU, then you need to install torch with CUDA with the GPU-enabled version.
Otherwise, the processor will be used.
### Download models:
```sh
python3 download_models.py
```

-----
## ⚙️ 2. Edit `config.py` (can skip)  

* DEVICE: if you have multiple GPUs, set device num which you want to use (or set 'cpu', but it's too slow).  
* PERSON_CONF: confidence/threshold for object detection (Yolo).  
* KEYPOINTS: it's my keypoints list, some of which uses to filter object bboxes by visibility (for example, if confidence of few keypoints < KPTS_CONF, we ignore that object). 
* KPTS_CONF: confidence of keypoints (visibility) .
if you want to change keypoints used to evaluate visibility, you can fix it in  `pose-estimation.py`.
* IOU_THRESHOLD: when we check if new objects in frame, we check IOU between all the boxes found by Yolo and all the boxes found by the tracker, so if IOU < IOU_THRESHOLD, we check keypoints and if all is ok, it's new object which will be added.
* XMEM_CONFIG: very important for your current task. Experiment with parameters or use default settings.
* MAX_OBJECT_CNT: if you don't know value of object in your tasks, set this value very large.  
* YOLO_EVERY: check new objects in frame every N frames.  
* INFERENCE_SIZE: video or sequence of frames resolution.
-----

## 🚀 3. Run
### Tracking
You can simply run it on your video with command:
  ```sh
  python3 tracking.py --video_path=INPUT_VIDEO_PATH.mp4 --width=1280 \
--height=768 --frames_to_propagate=600 --output_video_path=RESULT_VIDEO_PATH.mp4 --device=0 \
--person_conf=0.6 --kpts_conf=0.4 --iou_thresh=0.15 --yolo_every=2 --output_path=OUTPUT_CSV_PATH.csv
  ```
  You can also set `frames_to_propagate`: num of frames, which you want to process.
  After that you can get output video with animations (detection, tracking results) and csv-file with all information about objects in every frame.
  
 ### Metrics counting
 I wrote **custom** Precision, Recall and F1Score calculation for tracking task. It compares bboxes positions and their ids.  
⚠️ Please use it with labels from CVAT dataset exporting (the structure is described below)  
   
 You can simply run it on your labeled video or frames with command:
  ```sh
  python3 metrics_counting.py --labels_dir=LABELS_DIR_PATH --width=1280 \
--height=768  --device=0 --person_conf=0.6 --kpts_conf=0.4\
 --iou_thresh=0.15 --print_every=10
  ```
  Note that structure of LABELS_DIR_PATH should be:  
  ~~~~
LABELS_DIR_PATH
     |- first_dir
         |- obj_train_data
             |- frame0.jpg
             |- frame0.txt
             |- frame1.jpg
             |- frame1.txt
             ...
     |- second_dir
     ...
~~~~
Example. My LABELS_DIR_PATH is `test_files`:  
<div style="display: flex; align-items: center; justify-content: space-around;">
  <img src="https://i.ibb.co/pwdWXGV/image.png" height="300">
  <img src="https://i.ibb.co/JnRKfn3/2023-12-26-16-23-40.png" height="300">
</div>


Labels: Yolo  
(directory with txt files corresponding to frames, format of example.txt:  
```
0 0.265682 0.430208 0.057479 0.279509  
1 0.483107 0.486296 0.069411 0.337759  
... 
5 0.743799 0.467407 0.060016 0.289593
```

-----
## 🎯 4. Use project for your custom tasks
It's simply to change [`pose-estimation.py`](https://github.com/licksylick/AutoTrackAnything/blob/53d85446b110eaea189def1d30f95593e07a555b/pose_estimation.py#L9) and use different detection model (or your custom trained model):
1. Change model loading
2. In `get_filtered_bboxes_by_confidence` method return list with bboxes from your model
3. Enjoy 😊

-----

## ⭐️ BibTex of AutoTrackAnything:
Please star and cite this repo if you find project useful!  

```
@software{AutoTrackAnything,
  author = {Roman Lyskov},
  title = {AutoTrackAnything},
  year = {2024},
  url = {https://github.com/licksylick/AutoTrackAnything},
  license = {MIT}
}
```

```
@inproceedings{cheng2022xmem,
  title={{XMem}: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model},
  author={Cheng, Ho Kei and Alexander G. Schwing},
  booktitle={ECCV},
  year={2022}
}
```

```
@article{mobile_sam,
  title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung-Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal={arXiv preprint arXiv:2306.14289},
  year={2023}
}
```

```
@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
```
