# configs/model_utils.py
import os
import torch


def load_checkpoint(checkpoint_dir, filename, device):
    """Load a checkpoint file. Returns the checkpoint dict or None."""
    path = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(path):
        print(f"⚠ Checkpoint not found: {path}")
        return None
    ckpt = torch.load(path, weights_only=False, map_location=device)
    print(f"✓ Loaded checkpoint: {filename}")
    return ckpt


def load_model_weights(model, checkpoint_dir, filename, device,
                       state_key='model_state_dict'):
    """Load weights into a model. Returns True if successful."""
    ckpt = load_checkpoint(checkpoint_dir, filename, device)
    if ckpt is None:
        return False
    model.load_state_dict(ckpt[state_key], strict=False)
    return True