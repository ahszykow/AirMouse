#!/usr/bin/env python3
# openpose_preprocess.py  — robust, CPU‑friendly, hand‑only

import os, glob, sys, time
import numpy as np, cv2

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k): return x

# ───────── paths ─────────
RHD_ROOT = "/workspace/RHD_published_v2"
IMG_GLOB = os.path.join(RHD_ROOT, "evaluation", "color", "*.png")
OUT_NPY  = "op_preds.npy"

# ───────── OpenPose set‑up ─────────
# ensure Python knows where to find the OpenPose bindings
sys.path.append('/openpose/build/python')
from openpose import pyopenpose as op

params = {
    # model data
    "model_folder"        : "/openpose/models/",
    # hand detector only
    "hand"                : "true",     # run hand network
    "hand_detector"       : "2",        # internal palm detector
    # tweak palm detector to increase recall on small images
    "hand_scale_number"   : 6,           # scan 6 scales
    "hand_scale_range"    : 0.4,         # ±40% scale range
    "hand_net_resolution" : "368x368",  # higher resolution input
    # skip other networks
    "body"                : "0",        # skip body network
    "face"                : "false",    # skip face network
    "disable_blending"    : "true",     # no rendering → faster
    "display"             : "0",
    "disable_multi_thread": "true"      # safer inside Docker
}
wrapper = op.WrapperPython()
wrapper.configure(params)
wrapper.start()

# ───────── run inference ─────────
img_paths = sorted(glob.glob(IMG_GLOB))
print("Images:", len(img_paths))

preds, good = [], 0
for p in tqdm(img_paths, desc="processing"):
    t0  = time.time()
    img = cv2.imread(p)
    img = cv2.resize(img, (656, 656)) 
    if img is None:
        preds.append(np.full((21, 2), np.nan, np.float32))
        continue

    datum             = op.Datum()
    datum.cvInputData = img
    # wrap in VectorDatum so C++ sees a std::vector<shared_ptr<Datum>>
    wrapper.emplaceAndPop(op.VectorDatum([datum]))

    hk = datum.handKeypoints  # [left, right]
    coords = None
    # Try right first, then left
    for hand in (1, 0):
        if hk and hk[hand] is not None and hk[hand].shape[0] > 0:
            coords = hk[hand][0, :, :2].astype(np.float32)
            good += 1
            break
    if coords is None:
        coords = np.full((21, 2), np.nan, np.float32)
    preds.append(coords)
    tqdm.write(f"{os.path.basename(p)}  {time.time()-t0:.2f}s")

# save predictions
preds = np.stack(preds)
np.save(OUT_NPY, preds)
print(f"\nSaved {OUT_NPY}  shape={preds.shape}  "
      f"NaNs={np.isnan(preds).sum()}  valid={good}/{len(preds)}")
