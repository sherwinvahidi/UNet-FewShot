"""
few_shot_sampler.py
-------------------
Episode-based data loading for k-shot brain tumor segmentation.

Usage:
    sampler = FewShotSampler(support_ids, query_ids, TRAIN_DATASET_PATH, k_shot=5)
    support_batch, query_batch = sampler.sample_episode()
"""

import os
import cv2
import copy
import torch
import random
import numpy as np
import nibabel as nib
from torch.nn import CrossEntropyLoss

try:
    from configs.metrics import dice_score, hd95_multiclass
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from configs.metrics import dice_score, hd95_multiclass

IMG_SIZE       = 128
VOLUME_SLICES  = 100
VOLUME_START   = 22


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_patient_slices(patient_id, data_path, slices=None):
    """
    Load all (or a subset of) 2-channel slices for one patient.
    Returns images (N,2,H,W) and masks (N,H,W) as numpy arrays.
    """
    path = os.path.join(data_path, patient_id)
    flair = nib.load(os.path.join(path, f'{patient_id}_flair.nii')).get_fdata()
    t1ce  = nib.load(os.path.join(path, f'{patient_id}_t1ce.nii')).get_fdata()
    seg   = nib.load(os.path.join(path, f'{patient_id}_seg.nii')).get_fdata()

    slice_indices = slices if slices is not None else range(VOLUME_SLICES)
    images, masks = [], []

    for s in slice_indices:
        sl = s + VOLUME_START
        fl  = cv2.resize(flair[:, :, sl], (IMG_SIZE, IMG_SIZE))
        t1  = cv2.resize(t1ce[:, :, sl],  (IMG_SIZE, IMG_SIZE))
        sg  = cv2.resize(seg[:, :, sl],    (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_NEAREST)
        img = np.stack([fl, t1], axis=0)  # (2, H, W)
        mx  = img.max()
        if mx > 0:
            img = img / mx
        sg[sg == 4] = 3                   # remap label 4 → 3
        images.append(img)
        masks.append(sg)

    return np.stack(images), np.stack(masks)   # (N,2,H,W), (N,H,W)


def has_tumor(mask):
    """True if the mask contains at least one non-background pixel."""
    return mask.max() > 0


# ── Core sampler ─────────────────────────────────────────────────────────────

class FewShotSampler:
    """
    Samples (support, query) episode pairs for k-shot evaluation.

    Parameters
    ----------
    support_ids : list[str]   Patient IDs used as the support pool
    query_ids   : list[str]   Patient IDs used as query pool
    data_path   : str
    k_shot      : int         Number of labeled support slices (per class present)
    n_query     : int         Number of query slices per episode
    tumor_only  : bool        If True, only pick slices that contain tumor
    """

    def __init__(self, support_ids, query_ids, data_path,
                 k_shot=5, n_query=16, tumor_only=True):
        self.support_ids = support_ids
        self.query_ids   = query_ids
        self.data_path   = data_path
        self.k_shot      = k_shot
        self.n_query     = n_query
        self.tumor_only  = tumor_only

        # Pre-index slices so episodes are fast to sample
        print("Indexing support patients …")
        self.support_index = self._build_index(support_ids)
        print("Indexing query patients …")
        self.query_index   = self._build_index(query_ids)
        print(f"  Support slices: {len(self.support_index)}")
        print(f"  Query  slices : {len(self.query_index)}")

    # ── private ──────────────────────────────────────────────────

    def _build_index(self, patient_ids):
        """Returns list of (image_2hw, mask_hw) for every valid slice."""
        index = []
        for pid in patient_ids:
            imgs, msks = load_patient_slices(pid, self.data_path)
            for img, msk in zip(imgs, msks):
                if self.tumor_only and not has_tumor(msk):
                    continue
                index.append((img, msk))
        return index

    def _to_tensors(self, items):
        imgs  = torch.tensor(np.stack([x[0] for x in items]), dtype=torch.float32)
        masks = torch.tensor(np.stack([x[1] for x in items]), dtype=torch.long)
        return {'image': imgs, 'mask': masks}

    # ── public ───────────────────────────────────────────────────

    def sample_episode(self):
        """
        Returns
        -------
        support : dict  {'image': (k, 2, H, W), 'mask': (k, H, W)}
        query   : dict  {'image': (n, 2, H, W), 'mask': (n, H, W)}
        """
        support_items = random.sample(self.support_index,
                                      min(self.k_shot, len(self.support_index)))
        query_items   = random.sample(self.query_index,
                                      min(self.n_query,  len(self.query_index)))
        return self._to_tensors(support_items), self._to_tensors(query_items)

    def iter_episodes(self, n_episodes=100):
        """Generator — yields (support, query) tuples."""
        for _ in range(n_episodes):
            yield self.sample_episode()


# ── k-shot fine-tuning baseline ──────────────────────────────────────────────

def kshot_finetune_eval(pretrained_model, sampler, device,
                        lr=1e-4, finetune_steps=10, n_episodes=50,
                        compute_hd95=False):
    """
    Baseline: for each episode, fine-tune a copy of the pretrained model
    on k support slices, then evaluate on query slices.
    Returns list of mean-tumor Dice scores (and optionally HD95).
    """
    ce = CrossEntropyLoss()
    episode_dice = []
    episode_hd95 = []

    for ep, (support, query) in enumerate(sampler.iter_episodes(n_episodes)):
        model = copy.deepcopy(pretrained_model).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr)

        # Fine-tune on support set
        model.train()
        s_img  = support['image'].to(device)
        s_mask = support['mask'].to(device)
        for _ in range(finetune_steps):
            opt.zero_grad()
            loss = ce(model(s_img), s_mask)
            loss.backward()
            opt.step()

        # Evaluate on query set
        model.eval()
        q_img  = query['image'].to(device)
        q_mask = query['mask'].to(device)
        with torch.no_grad():
            preds = torch.argmax(model(q_img), dim=1)

        dice = dice_score(preds, q_mask)
        episode_dice.append(dice['mean_tumor_dice'])

        if compute_hd95:
            hd = hd95_multiclass(preds, q_mask)
            episode_hd95.append(hd['mean_tumor_hd95'])

        if (ep + 1) % 10 == 0:
            msg = f"  Episode {ep+1}/{n_episodes} | Dice: {np.mean(episode_dice):.4f}"
            if compute_hd95:
                msg += f" | HD95: {np.nanmean(episode_hd95):.2f} px"
            print(msg)

    print(f"\nk={sampler.k_shot} shot | "
          f"Dice: {np.mean(episode_dice):.4f} ± {np.std(episode_dice):.4f}")
    if compute_hd95:
        print(f"  HD95: {np.nanmean(episode_hd95):.2f} ± {np.nanstd(episode_hd95):.2f} px")

    if compute_hd95:
        return episode_dice, episode_hd95
    return episode_dice