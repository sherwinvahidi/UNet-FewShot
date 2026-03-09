# configs/config.py
import torch

class Config:
    # Paths
    TRAIN_DATASET_PATH = "/Users/sherwinvahidimowlavi/Downloads/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
    CHECKPOINT_DIR = "./checkpoints"
    RESULTS_DIR = "./results"
    
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