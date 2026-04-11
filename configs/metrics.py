# metrics.py
import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff

# ── Dice per class ──────────────────────────────────────────────
def dice_score(pred, target, num_classes=4, smooth=1e-6):
    """
    pred:   (N, H, W) long tensor – predicted class indices
    target: (N, H, W) long tensor – ground truth class indices
    Returns dict with per-class Dice and mean tumor Dice (classes 1-3).
    """
    scores = {}
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        intersection = (p * t).sum()
        scores[f'dice_class_{c}'] = (
            (2 * intersection + smooth) / (p.sum() + t.sum() + smooth)
        ).item()
    
    # BraTS convention: mean over tumor classes only (1=NCR, 2=ED, 3=ET)
    scores['mean_tumor_dice'] = np.mean([scores[f'dice_class_{c}'] for c in range(1, 4)])
    return scores

# ── HD95 ────────────────────────────────────────────────────────
def hd95(pred_mask, target_mask):
    """Binary masks (H, W) as numpy arrays. Returns HD95 in pixels."""
    pred_pts   = np.argwhere(pred_mask)
    target_pts = np.argwhere(target_mask)
    
    if len(pred_pts) == 0 and len(target_pts) == 0:
        return 0.0
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return np.inf
        
    d1 = directed_hausdorff(pred_pts, target_pts)[0]
    d2 = directed_hausdorff(target_pts, pred_pts)[0]
    
    # Compute all distances for 95th percentile
    from scipy.spatial import cKDTree
    tree1 = cKDTree(pred_pts)
    tree2 = cKDTree(target_pts)
    d_fwd = tree2.query(pred_pts)[0]
    d_bwd = tree1.query(target_pts)[0]
    all_d = np.concatenate([d_fwd, d_bwd])
    
    return float(np.percentile(all_d, 95))

def hd95_multiclass(pred, target, num_classes=4):
    """Averages HD95 over tumor classes (1-3)."""
    scores = {}
    for c in range(1, num_classes):
        p = (pred == c).cpu().numpy()
        t = (target == c).cpu().numpy()
        # Average over batch
        batch_hd = [hd95(p[i], t[i]) for i in range(p.shape[0])]
        finite = [x for x in batch_hd if np.isfinite(x)]
        scores[f'hd95_class_{c}'] = np.mean(finite) if finite else np.nan
        
    scores['mean_tumor_hd95'] = np.mean([scores[f'hd95_class_{c}'] for c in range(1, 4)])
    return scores

# ── Full evaluation loop ────────────────────────────────────────
def evaluate(model, loader, device, num_classes=4):
    model.eval()
    all_dice   = {f'dice_class_{c}': [] for c in range(num_classes)}
    all_dice['mean_tumor_dice'] = []
    all_hd95   = {f'hd95_class_{c}': [] for c in range(1, num_classes)}
    all_hd95['mean_tumor_hd95'] = []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            masks  = batch['mask'].to(device)

            logits = model(images)                        # (N, C, H, W)
            preds  = torch.argmax(logits, dim=1)          # (N, H, W)

            d = dice_score(preds, masks, num_classes)
            for k, v in d.items():
                all_dice[k].append(v)

            h = hd95_multiclass(preds, masks, num_classes)
            for k, v in h.items():
                all_hd95[k].append(v)

    results = {}
    for k, v in all_dice.items():
        results[k] = np.mean(v)
    for k, v in all_hd95.items():
        results[k] = np.nanmean(v)
    return results