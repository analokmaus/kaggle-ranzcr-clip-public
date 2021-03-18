import torch
import torch.nn as nn
from timm_latest import create_model
from segmentation_models_pytorch.encoders._base import EncoderMixin


class ResNet200dEncoder(nn.Module, EncoderMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self._out_channels = [3, 64, 256, 512, 1024, 2048]
        self._depth = 5
        self._in_channels = 3
        self._m = create_model(
            model_name='resnet200d',
            in_chans=self._in_channels,
            pretrained=False
        )
        self._m.global_pool = nn.Identity()
        self._m.fc = nn.Identity()

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self._m.conv1, self._m.bn1, self._m.act1),
            nn.Sequential(self._m.maxpool, self._m.layer1),
            self._m.layer2,
            self._m.layer3,
            self._m.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        return features

    def load_state_dict(self, state_dict):
        state_dict.pop("fc.bias")
        state_dict.pop("fc.weight")
        self._m.load_state_dict(state_dict, strict=False)


''' Add the following block to somewhere in your script :)
smp.encoders.encoders["resnet200d"] = {
    "encoder": ResNet200dEncoder,
    "pretrained_settings": {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet200d_ra2-bdba9bf9.pth",
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": {}
}
'''
