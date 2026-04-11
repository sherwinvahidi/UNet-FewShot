# models/bu_net.py
import torch.nn as nn
import segmentation_models_pytorch as smp

class BUNet(nn.Module):
    def __init__(self, encoder_name='resnet34', in_channels=2, num_classes=4):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
 
    def forward(self, x):
        return self.unet(x)