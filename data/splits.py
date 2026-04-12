# data/splits.py
"""
Shared data splitting and loading utilities.
Ensures consistent 70/20/10 train/val/test splits across all notebooks.

Usage:
    from data.splits import get_loaders, get_patient_splits

    # Full pipeline (datasets + loaders)
    train_loader, val_loader, test_loader = get_loaders(Config)

    # Just the IDs (for few-shot sampler, etc.)
    train_ids, val_ids, test_ids = get_patient_splits(Config.TRAIN_DATASET_PATH)
"""
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data.dataset import BraTSDataset


def fix_brats355(data_path):
    """Fix the known BraTS20_Training_355 filename issue."""
    old = os.path.join(data_path, "BraTS20_Training_355/W39_1998.09.19_Segm.nii")
    new = os.path.join(data_path, "BraTS20_Training_355/BraTS20_Training_355_seg.nii")
    if os.path.exists(old):
        os.rename(old, new)
        print("✓ Fixed BraTS20_Training_355 filename")


def get_patient_splits(data_path, random_state=42):
    """
    Returns (train_ids, val_ids, test_ids) with 70/20/10 split.

    Uses a fixed random_state so splits are identical across notebooks.
    """
    fix_brats355(data_path)

    dirs = [f.path for f in os.scandir(data_path) if f.is_dir()]
    all_ids = [d[d.rfind('/') + 1:] for d in dirs]

    train_test_ids, val_ids = train_test_split(
        all_ids, test_size=0.2, random_state=random_state
    )
    train_ids, test_ids = train_test_split(
        train_test_ids, test_size=0.125, random_state=random_state
    )

    print(f"✓ Splits -> Train: {len(train_ids)}, "
          f"Val: {len(val_ids)}, Test: {len(test_ids)}")
    return train_ids, val_ids, test_ids


def get_datasets(config):
    """Create train/val/test BraTSDataset objects."""
    train_ids, val_ids, test_ids = get_patient_splits(config.TRAIN_DATASET_PATH)

    kwargs = dict(
        data_path=config.TRAIN_DATASET_PATH,
        img_size=config.IMG_SIZE,
        volume_slices=config.VOLUME_SLICES,
        volume_start=config.VOLUME_START_AT,
    )

    train_dataset = BraTSDataset(train_ids, **kwargs)
    val_dataset = BraTSDataset(val_ids, **kwargs)
    test_dataset = BraTSDataset(test_ids, **kwargs)

    return train_dataset, val_dataset, test_dataset


def get_loaders(config, batch_size=None):
    """Create train/val/test DataLoaders."""
    train_ds, val_ds, test_ds = get_datasets(config)
    bs = batch_size or config.BATCH_SIZE

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches:   {len(val_loader)}")
    print(f"✓ Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader