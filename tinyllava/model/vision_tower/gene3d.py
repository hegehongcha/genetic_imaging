from transformers import Dinov2Model, AutoImageProcessor
import torch.nn as nn
from . import register_vision_tower
from .base import VisionTower
from monai.networks.nets import resnet
import torch
import os


class Med3D_DINOV2_Model(nn.Module):
    def __init__(self, cfg, use_projection=True):
        super().__init__()

        # 1. Load Med3D backbone (ResNet18 3D version)
        self.med3d = resnet.resnet18(spatial_dims=3, n_input_channels=1, num_classes=1)
        self.med3d.fc = nn.Identity()  # Remove the final FC layer

        # 2. Load DINOv2 ViT model
        self.dinov2 = Dinov2Model(cfg)

        # Optional: project Med3D output to 3x224x224 image-like tensor
        self.use_projection = use_projection
        if use_projection:
            self.projector = nn.Sequential(
                nn.Linear(512, 3 * 224 * 224),  # You can change 224x224 to 512x512
                nn.Tanh()  # Optional activation
            )

    def forward(self, volume3d, **kwargs):
        """
        volume3d: (B, 1, D, H, W) tensor, e.g. (1, 1, 64, 128, 128)
        """
        # 1. Extract Med3D features (B, 512, 1, 1, 1)
        feat = self.med3d(volume3d)
        feat = feat.view(feat.size(0), -1)  # (B, 512)

        if self.use_projection:
            # 2. Project to (B, 3, 224, 224)
            img_like = self.projector(feat).view(-1, 3, 224, 224)
        else:
            raise NotImplementedError("Direct DINOv2 vector input not implemented")

        # 3. Feed into DINOv2
        image_features = self.dinov2(pixel_values=img_like, output_hidden_states=True)
        image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]
        if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
            image_features = image_features[:, 1:]
        elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")

        return image_features


@register_vision_tower('gene3d')
class Gene3d_Dinov2_VisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)

        self._vision_tower = Med3D_DINOV2_Model(cfg)

        # self._image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name_or_path)

    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = kwargs.pop('pretrained_vision_tower_path', None)
        if pretrained_vision_tower_path is None:
            self._vision_tower.dinov2 = self._vision_tower.dinov2.from_pretrained(vision_tower_name, **kwargs)
            print("Loading vision tower (ViT) from ", vision_tower_name)
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

    def forward(self, x, **kwargs):
        device = x.data.device
        self.to(device)
        return self._vision_tower(x, **kwargs)