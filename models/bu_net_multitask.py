# models/bu_net_multitask_working.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class BUNetMultiTask(nn.Module):
    """Working BU-Net with both segmentation and classification heads"""
    
    def __init__(self, encoder_name='resnet34', in_channels=2, 
                 num_classes=4, num_tumor_types=5):
        super().__init__()
        
        # Use SMP's Unet for segmentation
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
        
        # Classification head - takes the deepest features
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_tumor_types)
        )
        
    def forward(self, x):
        # Get segmentation output
        seg_output = self.unet(x)
        
        # Get encoder features for classification
        features = self.unet.encoder(x)
        deepest = features[-1]
        
        # Classification output
        cls_output = self.classification_head(deepest)
        
        return seg_output, cls_output
    
    def get_prototypes(self, support_images, support_masks):
        """Simple prototype extraction"""
        self.eval()
        prototypes = []
        
        with torch.no_grad():
            for i in range(len(support_images)):
                img = support_images[i].unsqueeze(0)
                features = self.unet.encoder(img)
                deepest = features[-1]
                
                # Simple global average pooling for prototype
                prototype = deepest.mean(dim=(2, 3)).squeeze(0)
                prototypes.append(prototype)
            
            final_prototype = torch.stack(prototypes).mean(dim=0)
            
        return final_prototype