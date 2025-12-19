from transformers import Dinov2Model, AutoImageProcessor
import torch.nn as nn
from . import register_vision_tower
from .base import VisionTower
from monai.networks.nets import resnet
import torch
import os


class Med3D_Model(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Load Med3D backbone (ResNet18 3D version)
        self.med3d = resnet.resnet18(spatial_dims=3, n_input_channels=1, num_classes=1)
        self.med3d.fc = nn.Linear(self.med3d.fc.in_features, 6)

    def forward(self, volume3d, return_features=True, **kwargs):
        """
        volume3d: (B, 1, D, H, W) tensor, e.g. (1, 1, 64, 128, 128)
        """
        x = self.med3d.conv1(volume3d)
        x = self.med3d.bn1(x)
        x = self.med3d.act(x)
        x = self.med3d.maxpool(x)

        x = self.med3d.layer1(x)
        x = self.med3d.layer2(x)
        x = self.med3d.layer3(x)
        x = self.med3d.layer4(x)

        x = self.med3d.avgpool(x)  # <<<<<< 这里是你要的特征
        x = x.view(x.size(0), -1)  # 展平成(batch_size, 512)

        if return_features:
            x = x.view(x.size(0), 1, -1)
            return x  # 直接输出avgpool后的hidden state

        x = self.med3d.fc(x)  # 分类
        return x


@register_vision_tower('med3d')
class Gene3dVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)

        self._vision_tower = Med3D_Model()

        # self._image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name_or_path)

    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = kwargs.pop('pretrained_vision_tower_path', None)
        if pretrained_vision_tower_path is None:
            # self._vision_tower.dinov2 = self._vision_tower.dinov2.from_pretrained(vision_tower_name, **kwargs)
            print("Loading vision tower (Med3D) from ", vision_tower_name)
            vision_tower_weights = torch.load(
                os.path.join("/home/httang/project/llava/TinyLLaVA_Factory-main/gene_checkpoints/5fold/2/gene3d_classfication_no_roi",
                             'best_model.pth'), map_location='cpu')
            self._vision_tower.load_state_dict(vision_tower_weights)
        else:  # nn.Module
            if pretrained_vision_tower_path is not None:
                # print(self._vision_tower.named_parameters())
                # print(111, self._vision_tower.state_dict().keys())
                vision_tower_weights = torch.load(os.path.join(pretrained_vision_tower_path, 'pytorch_model.bin'),
                                                  map_location='cpu')

                # print(222, vision_tower_weights.keys())
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

                new_vision_tower_weights = {}
                for k, v in vision_tower_weights.items():
                    # 只替换fc1和fc2等中的base_layer
                    if ".base_layer." in k:
                        new_k = k.replace(".base_layer", "")
                        new_vision_tower_weights[new_k] = v
                    else:
                        new_vision_tower_weights[k] = v
                self._vision_tower.load_state_dict(new_vision_tower_weights)
            print("Loading vision tower from ", pretrained_vision_tower_path)

    def forward(self, x, return_features=True, **kwargs):
        device = x.data.device
        self.to(device)
        return self._vision_tower(x, return_features=True)