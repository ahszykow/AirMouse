#!/usr/bin/env python3
import os, glob, cv2, numpy as np

# 1) MMDetection imports & registry
from mmengine.registry import init_default_scope
from mmdet.apis import init_detector, inference_detector

# 2) MMPose imports & registry
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
from tqdm import tqdm

# ── PATHS ──────────────────────────────────────────────────────────
RHD_ROOT        = '/Users/anthonyszykowny/Documents/Spring 2025 Classes/CSE 473/AirMouse/RHD_published_v2'
IMG_GLOB        = os.path.join(RHD_ROOT, 'evaluation', 'color', '*.png')

DET_CONFIG     = '/Users/anthonyszykowny/Documents/Spring 2025 Classes/CSE 473/AirMouse/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
DET_CHECKPOINT = '/Users/anthonyszykowny/Downloads/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
DEVICE          = 'cpu'   # or 'cuda:0' if you have GPU

POSE_CONFIG     = '/Users/anthonyszykowny/Documents/Spring 2025 Classes/CSE 473/AirMouse/mmpose/configs/hand_2d_keypoint/topdown_heatmap/coco_wholebody_hand/td-hm_litehrnet-w18_8xb32-210e_coco-wholebody-hand-256x256.py'
POSE_CHECKPOINT = '/Users/anthonyszykowny/Downloads/litehrnet_w18_coco_wholebody_hand_256x256-d6945e6a_20210908.pth'

OUT_NPY         = 'rhd_mmpose_preds.npy'

# ── 1) LOAD DETECTOR ─────────────────────────────────────────────────────────
init_default_scope('mmdet')
det_model = init_detector(DET_CONFIG, DET_CHECKPOINT, device=DEVICE)

# ── 2) LOAD POSE ─────────────────────────────────────────────────────────────
init_default_scope('mmpose')
register_all_modules()
pose_model = init_model(POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE)

# ── 3) RUN INFERENCE ─────────────────────────────────────────────────────────
all_preds = []
for img_path in tqdm(sorted(glob.glob(IMG_GLOB)),
                     desc="Processing RHD images", unit="img"):
    img = cv2.imread(img_path)

    # 1) Detect hands
    init_default_scope('mmdet')
    det_results = inference_detector(det_model, img)
    data_sample = det_results
    inst        = data_sample.pred_instances
    raw_bboxes  = inst.bboxes.cpu().numpy()
    raw_scores  = inst.scores.cpu().numpy()
    all_dets    = np.concatenate([raw_bboxes, raw_scores[:, None]], 1)
    bboxes      = all_dets[all_dets[:,4] > 0.5, :4]
    
    if bboxes.shape[0] == 0:
        # no detection → all NaNs
        all_preds.append(np.full((21,2), np.nan, np.float32))
        continue

    # 2) Pose
    init_default_scope('mmpose')
    results = inference_topdown(pose_model, img, bboxes)

    # 3) Extract keypoints, but guard on count
    # might be tensor or ndarray
    kpts_data = results[0].pred_instances.keypoints
    try:
        kpts = kpts_data.cpu().numpy()
    except AttributeError:
        kpts = np.asarray(kpts_data)

    if kpts.ndim != 2 or kpts.shape[0] != 21:
        # wrong shape → fill with NaNs
        coords = np.full((21,2), np.nan, np.float32)
    else:
        coords = kpts[:, :2].astype(np.float32)

    all_preds.append(coords)

preds = np.stack(all_preds)
np.save(OUT_NPY, preds)
print(f"Saved → {OUT_NPY}, shape={preds.shape}, NaNs={np.isnan(preds).sum()}")