#!/usr/bin/env python3
import numpy as np

gt = np.load("gt_uv.npy")          # (N,21,2) – in RHD order
mp = np.load("mp_preds.npy")       # (N,21,2) – MediaPipe order
op = np.load("op_preds.npy")       # (N,21,2) – MediaPipe order

# -------- reorder GT to MediaPipe joint layout --------
MP_ORDER = [0,4,3,2,1, 8,7,6,5, 12,11,10,9, 16,15,14,13, 20,19,18,17]
gt = gt[:, MP_ORDER, :]

def summarise(name, pred, gt):
    errs = np.linalg.norm(pred - gt, axis=2)      # (N,21)
    valid = ~np.isnan(errs)
    if valid.sum() == 0:
        print(f"{name}: no valid predictions"); return
    print(f"{name:8}  mean={np.nanmean(errs):7.2f}px"
          f"  std={np.nanstd(errs):7.2f}"
          f"  PCK@5={np.nanmean(errs[valid] < 5):.3f}")

print("gt shape:", gt.shape, " NaNs:", np.isnan(gt).sum())
print("mp shape:", mp.shape, " NaNs:", np.isnan(mp).sum())
print("op shape:", op.shape, " NaNs:", np.isnan(op).sum())

summarise("MediaPipe", mp, gt)
summarise("OpenPose",  op, gt)
