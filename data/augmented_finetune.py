# data/augmented_finetune.py
"""
K-shot fine-tuning with support set augmentation.
Compares: fine-tune on k real examples vs. fine-tune on k examples + augmentations.
"""
import copy
import torch
import numpy as np
import random
from torch.nn import CrossEntropyLoss
from configs.metrics import dice_score

def augment_batch(images, masks):
    """
    Apply random augmentations to support images/masks.
    Returns augmented copies (originals unchanged).

    images: (k, 2, H, W) tensor
    masks:  (k, H, W) tensor
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

        # Random intensity jitter (image only, not mask)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = img * factor

        aug_images.append(img)
        aug_masks.append(msk)

    return torch.stack(aug_images), torch.stack(aug_masks)


def kshot_augmented_finetune_eval(pretrained_model, sampler, device,
                                   lr=1e-4, finetune_steps=10,
                                   n_episodes=50, augment_factor=3):
    """
    Like kshot_finetune_eval but augments support set before fine-tuning.

    For each episode:
      1. Sample k support + n query
      2. Create (augment_factor) augmented copies of support set
      3. Fine-tune on original + augmented support (k * (1 + augment_factor) examples)
      4. Evaluate on query set
    """
    ce = CrossEntropyLoss()
    episode_dice = []

    for ep, (support, query) in enumerate(sampler.iter_episodes(n_episodes)):
        model = copy.deepcopy(pretrained_model).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        s_img = support['image'].to(device)
        s_mask = support['mask'].to(device)

        # Create augmented support set
        all_imgs = [s_img]
        all_masks = [s_mask]
        for _ in range(augment_factor):
            aug_img, aug_mask = augment_batch(s_img, s_mask)
            all_imgs.append(aug_img)
            all_masks.append(aug_mask)

        aug_s_img = torch.cat(all_imgs, dim=0)
        aug_s_mask = torch.cat(all_masks, dim=0)

        # Fine-tune on augmented support set
        model.train()
        for _ in range(finetune_steps):
            opt.zero_grad()
            loss = ce(model(aug_s_img), aug_s_mask)
            loss.backward()
            opt.step()

        # Evaluate on query set (no augmentation)
        model.eval()
        q_img = query['image'].to(device)
        q_mask = query['mask'].to(device)
        with torch.no_grad():
            preds = torch.argmax(model(q_img), dim=1)

        dice = dice_score(preds, q_mask)
        episode_dice.append(dice['mean_tumor_dice'])

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes} | "
                  f"Mean Dice: {np.mean(episode_dice):.4f}")

    print(f"\nk={sampler.k_shot} (augmented {augment_factor}x) | "
          f"Mean Dice: {np.mean(episode_dice):.4f} "
          f"± {np.std(episode_dice):.4f}")
    return episode_dice