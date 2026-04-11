# data/synthetic_tumor_generator.py
import numpy as np
import torch
import cv2
from scipy.ndimage import gaussian_filter
import random
from torch.utils.data import Dataset

class SyntheticTumorGenerator:
    """Generate synthetic tumors for few-shot meta-learning"""
    
    def __init__(self, img_size=128):
        self.img_size = img_size
        
    def generate_tumor_mask(self, shape='ellipsoid', size_range=(5, 30)):
        """Generate synthetic tumor mask with variations"""
        mask = np.zeros((self.img_size, self.img_size))
        
        center_x = random.randint(20, self.img_size-20)
        center_y = random.randint(20, self.img_size-20)
        
        if shape == 'ellipsoid':
            a = random.randint(*size_range)
            b = random.randint(*size_range)
            rotation = random.uniform(0, np.pi)
            
            y, x = np.ogrid[:self.img_size, :self.img_size]
            x_rot = (x-center_x)*np.cos(rotation) + (y-center_y)*np.sin(rotation)
            y_rot = -(x-center_x)*np.sin(rotation) + (y-center_y)*np.cos(rotation)
            
            mask_proto = (x_rot**2/a**2 + y_rot**2/b**2) <= 1
            
        elif shape == 'spiculated':
            base_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            num_spicules = random.randint(5, 12)
            
            cv2.circle(base_mask, (center_x, center_y), 
                      random.randint(8, 15), 1, -1)
            
            for _ in range(num_spicules):
                angle = random.uniform(0, 2*np.pi)
                length = random.randint(10, 20)
                end_x = int(center_x + length * np.cos(angle))
                end_y = int(center_y + length * np.sin(angle))
                end_x = np.clip(end_x, 0, self.img_size-1)
                end_y = np.clip(end_y, 0, self.img_size-1)
                cv2.line(base_mask, (center_x, center_y), 
                        (end_x, end_y), 1, random.randint(3, 6))
            
            mask_proto = base_mask > 0
            
        elif shape == 'multifocal':
            mask_proto = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            num_foci = random.randint(2, 5)
            for _ in range(num_foci):
                cx = np.clip(center_x + random.randint(-20, 20), 0, self.img_size-1)
                cy = np.clip(center_y + random.randint(-20, 20), 0, self.img_size-1)
                radius = random.randint(5, 12)
                cv2.circle(mask_proto, (cx, cy), radius, 1, -1)
        else:
            mask_proto = np.zeros((self.img_size, self.img_size))
        
        # Add boundary irregularity
        if random.random() > 0.5:
            noise = np.random.randn(self.img_size, self.img_size) * 0.1
            mask_proto = (mask_proto * (1 + noise)) > 0.5
            
        return mask_proto.astype(np.float32)
    
    def apply_tumor_to_healthy(self, healthy_image, tumor_mask, 
                               intensity_factor=(1.2, 1.8)):
        """Apply synthetic tumor to healthy brain tissue"""
        factor = random.uniform(*intensity_factor)
        tumor_region = tumor_mask * healthy_image * factor
        
        # Add noise for texture variation
        noise = np.random.randn(self.img_size, self.img_size) * 0.05
        tumor_region = tumor_region * (1 + noise)
        
        result = healthy_image * (1 - tumor_mask) + tumor_region
        return result, tumor_mask
    
    def smooth_mask(self, mask, sigma=1.0):
        """Smooth tumor boundaries for realism"""
        smoothed = gaussian_filter(mask.astype(float), sigma=sigma)
        return (smoothed > 0.5).astype(np.float32)
    
    def generate_augmented_sample(self, healthy_image, 
                                   shape=None, size_range=(5, 30)):
        """Generate one complete synthetic tumor sample"""
        shapes = ['ellipsoid', 'spiculated', 'multifocal']
        if shape is None:
            shape = random.choice(shapes)
            
        mask = self.generate_tumor_mask(shape=shape, size_range=size_range)
        mask = self.smooth_mask(mask)
        image, mask = self.apply_tumor_to_healthy(healthy_image, mask)
        
        return image, mask


class SyntheticAugmentedDataset(Dataset):
    """
    Dataset that mixes real BraTS data with synthetic tumors
    """
    
    def __init__(self, real_dataset, synthetic_ratio=0.3, img_size=128):
        self.real_dataset = real_dataset
        self.synthetic_ratio = synthetic_ratio
        self.generator = SyntheticTumorGenerator(img_size)
        self.img_size = img_size
        
    def __len__(self):
        return len(self.real_dataset)
    
    def __getitem__(self, idx):
        # Get real sample first
        real_sample = self.real_dataset[idx]
        
        # Decide whether to use synthetic
        if random.random() > self.synthetic_ratio:
            # Return real sample as-is
            return real_sample
        
        # Generate synthetic version
        image = real_sample['image'].numpy()  # (2, H, W)
        
        # Apply synthetic tumor to FLAIR channel
        flair = image[0]
        shape = random.choice(['ellipsoid', 'spiculated', 'multifocal'])
        
        aug_flair, tumor_mask = self.generator.generate_augmented_sample(
            flair, shape=shape
        )
        
        # Create augmented image
        aug_image = np.stack([aug_flair, image[1]], axis=0)
        
        # Create binary mask
        aug_mask = (tumor_mask > 0.5).astype(np.int64)
        
        return {
            'image': torch.from_numpy(aug_image).float(),
            'mask': torch.from_numpy(aug_mask).long(),
            'patient_id': real_sample['patient_id'] + '_syn',
            'slice_idx': real_sample['slice_idx'],
            'synthetic': True
        }