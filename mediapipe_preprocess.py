#!/usr/bin/env python3
"""
Run MediaPipe‑Hands on every RHD evaluation image and save:

    gt_uv.npy   – (N,21,2) ground‑truth in MediaPipe joint order, pixels
    mp_preds.npy – (N,21,2) MediaPipe predictions, pixels
"""
import os, glob, pickle, cv2, numpy as np, mediapipe as mp

# ---------------- paths ----------------
RHD_ROOT     = "/Users/anthonyszykowny/Documents/Spring 2025 Classes/CSE 473/AirMouse/RHD_published_v2"
ANN_PICKLE   = os.path.join(RHD_ROOT, "evaluation", "anno_evaluation.pickle")
IMG_GLOB     = os.path.join(RHD_ROOT, "evaluation", "color", "*.png")
GT_OUT       = "gt_uv.npy"
MP_OUT       = "mp_preds.npy"

# -------------- load GT ---------------
with open(ANN_PICKLE, "rb") as f:
    ann = pickle.load(f)

# RHD right‑hand joints 21‑41 → MediaPipe order
MP_ORDER = [0,4,3,2,1, 8,7,6,5, 12,11,10,9,
            16,15,14,13, 20,19,18,17]

img_paths = sorted(glob.glob(IMG_GLOB))
gt_uv = np.zeros((len(img_paths), 21, 2), np.float32)

for i, path in enumerate(img_paths):
    data   = ann[i]
    uv_vis = np.asarray(data["uv_vis"], dtype=np.float32)   # (42,3)
    uv     = uv_vis[:, :2]                                  # (42,2) pixels already
    gt_uv[i] = uv[21: , :][MP_ORDER]                        # reorder

np.save(GT_OUT, gt_uv)
print(f"✔ Saved ground truth → {GT_OUT},  shape {gt_uv.shape}")

# -------------- MediaPipe --------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

mp_preds = []

for p in img_paths:
    img = cv2.imread(p)
    h, w = img.shape[:2]
    res  = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:
        lm   = res.multi_hand_landmarks[0].landmark
        pts  = np.array([[p_.x * w, p_.y * h] for p_ in lm], dtype=np.float32)
    else:
        pts  = np.full((21, 2), np.nan, np.float32)

    mp_preds.append(pts)

hands.close()
mp_preds = np.stack(mp_preds)
np.save(MP_OUT, mp_preds)
print(f"✔ Saved MediaPipe preds → {MP_OUT},  shape {mp_preds.shape},  NaNs {np.isnan(mp_preds).sum()}")