# models/prototypical_segmentation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class PrototypicalSegmentation(nn.Module):
    """
    Prototypical Networks adapted for segmentation
    
    Key idea: Learn prototypes for each class from support set,
    then segment query images by pixel-wise similarity to prototypes
    """
    
    def __init__(self, encoder_name='resnet34', in_channels=2, num_classes=4):
        super().__init__()
        
        # Use encoder from U-Net
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
        
        self.encoder = self.unet.encoder
        self.decoder = self.unet.decoder
        self.seg_head = self.unet.segmentation_head
        
        # Get encoder output channels
        self.encoder_channels = self.encoder.out_channels[-1]
        
    def extract_features(self, x):
        """Extract deep features from encoder"""
        features = self.encoder(x)
        return features
    
    def compute_prototypes(self, support_images, support_masks, num_classes=4):
        """
        Improved prototype computation with per-class weighting
        """
        self.eval()
        prototypes = []
        
        with torch.no_grad():
            # Extract features for all support images
            all_features = []
            for img in support_images:
                feats = self.extract_features(img.unsqueeze(0))
                deep_feat = feats[-1].squeeze(0)  # (C, H_f, W_f)
                all_features.append(deep_feat)
            
            # Compute prototype for each class
            for class_id in range(num_classes):
                class_prototypes = []
                class_weights = []  # Track how much each sample contributes
                
                for feat, mask in zip(all_features, support_masks):
                    # Resize mask to match feature map
                    mask_resized = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=feat.shape[-2:],
                        mode='nearest'
                    ).squeeze()
                    
                    # Get pixels belonging to this class
                    class_mask = (mask_resized == class_id).float()
                    
                    # Compute weighted average (weight by number of pixels)
                    if class_mask.sum() > 0:
                        masked_feat = feat * class_mask.unsqueeze(0)
                        proto = masked_feat.sum(dim=(1, 2)) / class_mask.sum()
                        class_prototypes.append(proto)
                        class_weights.append(class_mask.sum())  # Weight by area
                
                # Weighted average of prototypes (larger tumor regions = more weight)
                if len(class_prototypes) > 0:
                    weights = torch.tensor(class_weights).to(support_images.device)
                    weights = weights / weights.sum()  # Normalize
                    
                    weighted_proto = sum(w * p for w, p in zip(weights, class_prototypes))
                    prototypes.append(weighted_proto)
                else:
                    # Fallback: zero prototype
                    prototypes.append(torch.zeros(self.encoder_channels, 
                                                device=support_images.device))
        
        return prototypes
    
    def predict_with_prototypes(self, query_images, prototypes, training=False):
        # Use context manager based on training flag
        if training:
            # Allow gradients
            query_features = self.extract_features(query_images)
            deep_feat = query_features[-1]
        else:
            # No gradients
            with torch.no_grad():
                query_features = self.extract_features(query_images)
                deep_feat = query_features[-1]
        
        # Rest of the method stays the same
        B, C, H_f, W_f = deep_feat.shape
        
        distance_maps = []
        for proto in prototypes:
            proto_expanded = proto.view(1, C, 1, 1).expand(B, C, H_f, W_f)
            dist = -torch.norm(deep_feat - proto_expanded, dim=1)
            distance_maps.append(dist)
        
        distance_tensor = torch.stack(distance_maps, dim=1)
        
        seg_output = F.interpolate(
            distance_tensor,
            size=(query_images.shape[2], query_images.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        
        return seg_output
    
    def forward(self, x):
        """Standard forward pass (non-episodic)"""
        return self.unet(x)
    
    def forward_with_prototype_attention(self, query_images, support_images, support_masks):
        """
        Use prototypes to guide segmentation via attention
        """
        # Extract prototypes (no grad)
        with torch.no_grad():
            prototypes = self.compute_prototypes(support_images, support_masks)
        
        # Extract query features
        query_features = self.extract_features(query_images)
        deep_feat = query_features[-1]  # (B, C, H_f, W_f)
        
        B, C, H_f, W_f = deep_feat.shape
        
        # Compute similarity to each prototype
        similarity_maps = []
        for proto in prototypes:
            # Expand prototype: (C,) -> (1, C, 1, 1) -> (B, C, H_f, W_f)
            proto_expanded = proto.view(1, C, 1, 1).expand(B, C, H_f, W_f)
            
            # Cosine similarity per spatial location
            similarity = F.cosine_similarity(deep_feat, proto_expanded, dim=1)  # (B, H_f, W_f)
            similarity_maps.append(similarity)
        
        # Stack: (B, num_classes, H_f, W_f)
        attention = torch.stack(similarity_maps, dim=1)
        
        # Upsample attention to image size
        attention_upsampled = F.interpolate(
            attention,
            size=(query_images.shape[2], query_images.shape[3]),  # (128, 128)
            mode='bilinear',
            align_corners=False
        )
        
        # Softmax to get attention weights
        attention_weights = attention_upsampled.softmax(dim=1)  # (B, num_classes, 128, 128)
        
        # Standard U-Net segmentation
        seg_output = self.unet(query_images)  # (B, num_classes, 128, 128)
        
        # Apply attention weighting
        attended_output = seg_output * (1 + attention_weights)  # Boost by attention
        
        return attended_output