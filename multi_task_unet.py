import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskUNet(nn.Module):
    def __init__(self, base_unet, num_classes):
        super(MultiTaskUNet, self).__init__()
        assert base_unet is not None, "Base U-Net cannot be None."
        self.encoder = base_unet.encoder  # Shared encoder
        self.segmentation_head = base_unet.decoder  # Segmentation head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_unet.encoder.output_channels[-1], num_classes)
        )

        self.decoder = self.segmentation_head

    def forward(self, x):
        features = self.encoder(x)  # Shared encoder
        segmentation_output = self.segmentation_head(features)  # Segmentation
        classification_output = self.classification_head(features[-1])  # Classification
        return segmentation_output, classification_output
