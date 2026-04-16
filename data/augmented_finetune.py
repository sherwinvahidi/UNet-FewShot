# data/augmented_finetune.py
"""
Support set augmentation for few-shot evaluation.

Provides augmentation utilities and augmented evaluation functions
for all three methods: fine-tuning baseline, prototypical networks, and MAML.
"""
import copy
import torch
import numpy as np
import random
from torch.nn import CrossEntropyLoss
import segmentation_models_pytorch as smp

try:
    from configs.metrics import dice_score
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from configs.metrics import dice_score


# ── Augmentation Transforms ──────────────────────────────────────────────

def augment_batch(images, masks):
    """
    Apply random spatial + intensity augmentations to support images/masks.
    Returns augmented copies (originals unchanged).

    Args:
        images: (k, 2, H, W) tensor
        masks:  (k, H, W) tensor

    Returns:
        aug_images: (k, 2, H, W) tensor
        aug_masks:  (k, H, W) tensor
    """
    aug_images, aug_masks = [], []

    for img, msk in zip(images, masks):
        # Random horizontal flip
        if random.random() > 0.5:
            img = torch.flip(img, dims=[-1])
            msk = torch.flip(msk, dims=[-1])

        # Random vertical flip
        if random.random() > 0.5:
            img = torch.flip(img, dims=[-2])
            msk = torch.flip(msk, dims=[-2])

        # Random 90-degree rotation
        k_rot = random.choice([0, 1, 2, 3])
        if k_rot > 0:
            img = torch.rot90(img, k_rot, dims=[-2, -1])
            msk = torch.rot90(msk, k_rot, dims=[-2, -1])

        # Random intensity jitter (image only)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = img * factor

        aug_images.append(img)
        aug_masks.append(msk)

    return torch.stack(aug_images), torch.stack(aug_masks)


def build_augmented_support(s_img, s_mask, augment_factor=3):
    """
    Create an augmented support set: original + augment_factor copies.

    Args:
        s_img:  (k, 2, H, W) tensor on device
        s_mask: (k, H, W) tensor on device
        augment_factor: number of augmented copies to add

    Returns:
        aug_img:  (k * (1 + augment_factor), 2, H, W)
        aug_mask: (k * (1 + augment_factor), H, W)
    """
    all_imgs = [s_img]
    all_masks = [s_mask]

    for _ in range(augment_factor):
        a_img, a_mask = augment_batch(s_img, s_mask)
        all_imgs.append(a_img)
        all_masks.append(a_mask)

    return torch.cat(all_imgs, dim=0), torch.cat(all_masks, dim=0)


# ── Method 1: Augmented Fine-Tuning Baseline ────────────────────────────

def kshot_augmented_finetune_eval(pretrained_model, sampler, device,
                                   lr=1e-4, finetune_steps=10,
                                   n_episodes=50, augment_factor=3,
                                   compute_hd95=False):
    from configs.metrics import hd95_multiclass

    ce = CrossEntropyLoss()
    episode_dice = []
    episode_hd95 = []

    for ep, (support, query) in enumerate(sampler.iter_episodes(n_episodes)):
        model = copy.deepcopy(pretrained_model).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        s_img = support['image'].to(device)
        s_mask = support['mask'].to(device)

        aug_img, aug_mask = build_augmented_support(s_img, s_mask, augment_factor)

        model.train()
        for _ in range(finetune_steps):
            opt.zero_grad()
            loss = ce(model(aug_img), aug_mask)
            loss.backward()
            opt.step()

        model.eval()
        q_img = query['image'].to(device)
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

    print(f"\nk={sampler.k_shot} (aug {augment_factor}x) | "
          f"Dice: {np.mean(episode_dice):.4f} ± {np.std(episode_dice):.4f}")
    if compute_hd95:
        print(f"  HD95: {np.nanmean(episode_hd95):.2f} ± {np.nanstd(episode_hd95):.2f} px")
        return episode_dice, episode_hd95
    return episode_dice


# ── Method 2: Augmented Prototypical Networks ────────────────────────────

def kshot_augmented_prototypical_eval(proto_model, sampler, device,
                                      n_episodes=50, augment_factor=3,
                                      compute_hd95=False):
    from configs.metrics import hd95_multiclass

    loss_fn = smp.losses.DiceLoss(mode='multiclass')
    episode_dice = []
    episode_hd95 = []

    proto_model.eval()

    for ep, (support, query) in enumerate(sampler.iter_episodes(n_episodes)):
        s_img = support['image'].to(device)
        s_mask = support['mask'].to(device)
        q_img = query['image'].to(device)
        q_mask = query['mask'].to(device)

        aug_img, aug_mask = build_augmented_support(s_img, s_mask, augment_factor)

        with torch.no_grad():
            query_pred = proto_model.forward_with_prototype_attention(
                q_img, aug_img, aug_mask
            )
            loss = loss_fn(query_pred, q_mask)
            dice = 1 - loss.item()
            episode_dice.append(dice)

            if compute_hd95:
                preds = torch.argmax(query_pred, dim=1)
                hd = hd95_multiclass(preds, q_mask)
                episode_hd95.append(hd['mean_tumor_hd95'])

        if (ep + 1) % 10 == 0:
            msg = f"  Episode {ep+1}/{n_episodes} | Dice: {np.mean(episode_dice):.4f}"
            if compute_hd95:
                msg += f" | HD95: {np.nanmean(episode_hd95):.2f} px"
            print(msg)

    print(f"\nProto k={sampler.k_shot} (aug {augment_factor}x) | "
          f"Dice: {np.mean(episode_dice):.4f} ± {np.std(episode_dice):.4f}")
    if compute_hd95:
        print(f"  HD95: {np.nanmean(episode_hd95):.2f} ± {np.nanstd(episode_hd95):.2f} px")
        return episode_dice, episode_hd95
    return episode_dice


# ── Method 3: Augmented MAML ────────────────────────────────────────────

def kshot_augmented_maml_eval(maml_model, sampler, device,
                               inner_lr=0.01, num_inner_steps=5,
                               n_episodes=50, augment_factor=3,
                               compute_hd95=False):
    from configs.metrics import hd95_multiclass

    loss_fn = smp.losses.DiceLoss(mode='multiclass')
    episode_dice = []
    episode_hd95 = []

    for ep, (support, query) in enumerate(sampler.iter_episodes(n_episodes)):
        s_img = support['image'].to(device)
        s_mask = support['mask'].to(device)
        q_img = query['image'].to(device)
        q_mask = query['mask'].to(device)

        aug_img, aug_mask = build_augmented_support(s_img, s_mask, augment_factor)

        adapted_model = copy.deepcopy(maml_model.model).to(device)
        adapted_model.train()
        inner_opt = torch.optim.SGD(adapted_model.parameters(), lr=inner_lr)

        for _ in range(num_inner_steps):
            inner_opt.zero_grad()
            pred = adapted_model(aug_img)
            loss = loss_fn(pred, aug_mask)
            loss.backward()
            inner_opt.step()

        adapted_model.eval()
        with torch.no_grad():
            q_pred = adapted_model(q_img)
            q_loss = loss_fn(q_pred, q_mask)
            dice = 1 - q_loss.item()
            episode_dice.append(dice)

            if compute_hd95:
                preds = torch.argmax(q_pred, dim=1)
                hd = hd95_multiclass(preds, q_mask)
                episode_hd95.append(hd['mean_tumor_hd95'])

        if (ep + 1) % 10 == 0:
            msg = f"  Episode {ep+1}/{n_episodes} | Dice: {np.mean(episode_dice):.4f}"
            if compute_hd95:
                msg += f" | HD95: {np.nanmean(episode_hd95):.2f} px"
            print(msg)

    print(f"\nMAML k={sampler.k_shot} (aug {augment_factor}x) | "
          f"Dice: {np.mean(episode_dice):.4f} ± {np.std(episode_dice):.4f}")
    if compute_hd95:
        print(f"  HD95: {np.nanmean(episode_hd95):.2f} ± {np.nanstd(episode_hd95):.2f} px")
        return episode_dice, episode_hd95
    return episode_dice