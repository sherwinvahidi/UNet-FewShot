import torch
import os
import kagglehub

def get_kaggle_dataset_path():
    print("Verifying BraTS2020 dataset via KaggleHub...")
    # Downloads once and caches locally. Subsequent calls are instant.
    kaggle_path = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation")
    
    train_path = None
    # Walk the directory to find the actual training data folder
    for root, dirs, files in os.walk(kaggle_path):
        if "MICCAI_BraTS2020_TrainingData" in dirs:
            train_path = os.path.join(root, "MICCAI_BraTS2020_TrainingData")
            break
            
    # Fallback to common structures just in case
    if train_path is None:
        candidates = [
            os.path.join(kaggle_path, "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData"),
            os.path.join(kaggle_path, "MICCAI_BraTS2020_TrainingData"),
            kaggle_path,
        ]
        for c in candidates:
            if os.path.exists(c):
                train_path = c
                break
                
    assert train_path is not None, f"Could not locate MICCAI_BraTS2020_TrainingData inside {kaggle_path}."
    return train_path

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CONFIG_DIR)

class Config:
    # Dynamic Path Setup
    TRAIN_DATASET_PATH = get_kaggle_dataset_path()
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
    RESULTS_DIR = os.path.join(ROOT_DIR, "results")
    
    # Data
    IMG_SIZE = 128
    VOLUME_SLICES = 100
    VOLUME_START_AT = 22
    NUM_CLASSES = 4
    
    # Training
    BATCH_SIZE = 8
    NUM_EPOCHS = 35
    LEARNING_RATE = 0.001
    
    # Few-shot
    K_SHOT_VALUES = [1, 5, 10, 20]
    N_QUERY = 10
    
    # Model
    ENCODER_NAME = 'resnet34'
    IN_CHANNELS = 2
    NUM_TUMOR_TYPES = 5
    
    # Device
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Loss weights
    SEG_LOSS_WEIGHT = 1.0
    CLS_LOSS_WEIGHT = 0.5
    
    @classmethod
    def create_dirs(cls):
        import os
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)