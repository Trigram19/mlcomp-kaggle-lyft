import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from catalyst.contrib import registry


@registry.Model
class LyftModel(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.backbone._conv_stem = nn.Conv2d(
        5, 32, kernel_size=(3, 3), stride=(2, 2), bias=False
      )

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        self.future_len = cfg["model_params"]["future_num_frames"]
        backbone_out_features = 512
        num_targets = 2 * cfg["model_params"]["future_num_frames"]
        self.head = nn.Sequential(
            nn.Linear(in_features=1000, out_features=4096),
        )
        self.num_preds = num_targets * 3
        self.num_modes = 3
        self.logit = nn.Linear(4096, out_features=self.num_preds + 3)
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        x = self.logit(x)
        x = x.view(12, 50, 2)
        return x
