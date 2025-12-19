from transformers import Dinov2Model, AutoImageProcessor
import torch.nn as nn
from . import register_vision_tower
from .base import VisionTower
from monai.networks.nets import resnet
import torch
import os


class Med3D_ROI_Model(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Load Med3D backbone (ResNet18 3D version)
        self.med3d = resnet.resnet18(spatial_dims=3, n_input_channels=1, num_classes=1)
        self.med3d.fc = nn.Linear(self.med3d.fc.in_features, 6)

    def forward(self, volume3d, return_features=True, **kwargs):
        """
        volume3d: (B, 1, D, H, W) tensor, e.g. (1, 1, 64, 128, 128)
        """
#         image_features_list = []
        # print("volume3d: ", volume3d.shape)
#         with torch.no_grad():
#             for v in volume3d:
#                 x = self.med3d.conv1(v)
#                 x = self.med3d.bn1(x)
#                 x = self.med3d.act(x)
#                 x = self.med3d.maxpool(x)

#                 x = self.med3d.layer1(x)
#                 x = self.med3d.layer2(x)
#                 x = self.med3d.layer3(x)
#                 x = self.med3d.layer4(x)

#                 x = self.med3d.avgpool(x)  # <<<<<< 这里是你要的特征
#                 x = x.view(x.size(0), -1)  # 展平成(batch_size, 512)
#                 image_features_list.append(x)
#             image_features = torch.stack(image_features_list)
#         del volume3d
#         del image_features_list  
        image_features = volume3d
        return image_features
    
    def encode(self, volume3d, return_features=True, **kwargs):
        """
        volume3d: (B, 1, D, H, W) tensor, e.g. (1, 1, 64, 128, 128)
        return_features: if True, returns a list of all intermediate features
        """
        features = []
        volume3d = volume3d.to(dtype=next(self.parameters()).dtype)
        x = self.med3d.conv1(volume3d)
        features.append(x)
        x = self.med3d.bn1(x)
        x = self.med3d.act(x)
        features.append(x)
        x = self.med3d.maxpool(x)
        features.append(x)  # After stem

        x = self.med3d.layer1(x)
        features.append(x)  # After layer1

        x = self.med3d.layer2(x)
        features.append(x)  # After layer2

        x = self.med3d.layer3(x)
        features.append(x)  # After layer3

        x = self.med3d.layer4(x)
        features.append(x)  # After layer4

        x = self.med3d.avgpool(x)
        features.append(x)  # After avgpool

        return features  # Only return final pooled feature


@register_vision_tower('medroi3d')
class GeneROI3dVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)

        self._vision_tower = Med3D_ROI_Model()

        # self._image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name_or_path)

    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = kwargs.pop('pretrained_vision_tower_path', None)
        if pretrained_vision_tower_path is None:
            # self._vision_tower.dinov2 = self._vision_tower.dinov2.from_pretrained(vision_tower_name, **kwargs)
            print("Loading vision tower (Med3D) from ", vision_tower_name)
            vision_tower_weights = torch.load(
                os.path.join("/home/httang/project/llava/TinyLLaVA_Factory-main/gene_checkpoints/gene3d_classfication",
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
    
    def encode(self, x, return_features=True, **kwargs):
        device = x.data.device
        self.to(device)
        return self._vision_tower.encode(x, return_features=True)
    