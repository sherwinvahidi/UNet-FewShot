import torch
from torch.utils.data import Dataset
import nibabel as nib
import cv2
import numpy as np
import os

class BraTSDataset(Dataset):
    def __init__(self, patient_ids, data_path, img_size=128, 
                 volume_slices=100, volume_start=22):
        self.patient_ids = patient_ids
        self.data_path = data_path
        self.img_size = img_size
        self.volume_slices = volume_slices
        self.volume_start = volume_start
        
        # Validate files exist
        self.valid_ids = self._validate_patients()
        
    def _validate_patients(self):
        valid = []
        for pid in self.patient_ids:
            patient_path = os.path.join(self.data_path, pid)
            required_files = [
                f'{pid}_flair.nii',
                f'{pid}_t1ce.nii',
                f'{pid}_seg.nii'
            ]
            if all(os.path.exists(os.path.join(patient_path, f)) for f in required_files):
                valid.append(pid)
            else:
                print(f"⚠ Skipping {pid}: missing files")
        print(f"✓ Valid patients: {len(valid)}/{len(self.patient_ids)}")
        return valid
        
    def __len__(self):
        return len(self.valid_ids) * self.volume_slices
    
    def __getitem__(self, idx):
        patient_idx = idx // self.volume_slices
        slice_idx = idx % self.volume_slices
        
        patient_id = self.valid_ids[patient_idx]
        patient_path = os.path.join(self.data_path, patient_id)
        
        # Load data
        flair = nib.load(os.path.join(patient_path, f'{patient_id}_flair.nii')).get_fdata()
        t1ce = nib.load(os.path.join(patient_path, f'{patient_id}_t1ce.nii')).get_fdata()
        seg = nib.load(os.path.join(patient_path, f'{patient_id}_seg.nii')).get_fdata()
        
        # Extract slice
        slice_num = slice_idx + self.volume_start
        flair_slice = cv2.resize(flair[:, :, slice_num], (self.img_size, self.img_size))
        t1ce_slice = cv2.resize(t1ce[:, :, slice_num], (self.img_size, self.img_size))
        seg_slice = cv2.resize(seg[:, :, slice_num], (self.img_size, self.img_size),
                            interpolation=cv2.INTER_NEAREST)
        
        image = np.stack([flair_slice, t1ce_slice], axis=0)
        seg_slice[seg_slice == 4] = 3
        image = image / np.max(image) if np.max(image) > 0 else image
        
        return {
            'image': torch.from_numpy(image).float(),
            'mask': torch.from_numpy(seg_slice).long(),
            'patient_id': patient_id,
            'slice_idx': slice_idx,
            'synthetic': False
        }