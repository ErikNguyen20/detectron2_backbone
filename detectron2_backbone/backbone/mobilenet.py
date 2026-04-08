import torch
from torch import nn
from torch.nn import BatchNorm2d

import fvcore.nn.weight_init as weight_init

from detectron2.layers import Conv2d, FrozenBatchNorm2d, ShapeSpec
from detectron2.modeling import Backbone, BACKBONE_REGISTRY, FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool, LastLevelP6P7

from torchvision import models

__all__ = [
    "MobileNetV3Large",
    "build_mobilenet_v3_large_fpn_backbone",
    "build_retina_mobilenet_v3_large_fpn_backbone"
]

MobileNetV3_cfg = {
    "mobilenet_v3_large": {
        "out_feature_channels": [24, 40, 112, 960],
        "model": models.mobilenet_v3_large,
        "weights": models.MobileNet_V3_Large_Weights.DEFAULT,
    }
}


class MobileNetV3Large(Backbone):
    """
    Should freeze bn
    """
    def __init__(self, cfg, model_name="mobilenet_v3_large", freeze_at=0):
        super(MobileNetV3Large, self).__init__()
        self._out_features = ["res2", "res3", "res4", "res5"]  # cfg.MODEL.RESNETS.OUT_FEATURES

        if model_name not in MobileNetV3_cfg:
            raise ValueError("Invalid MobileNetV3 model")
        else:
            weights = MobileNetV3_cfg[model_name]["weights"]
            model = MobileNetV3_cfg[model_name]["model"](weights=weights)

        # torchvision MobileNetV3 backbone
        self.features = model.features

        # res2/res3/res4/res5 feature indices in torchvision MobileNetV3-Large
        self.return_features_indices = [3, 6, 12, 16]

        # Freeze all BN layers
        self.features = FrozenBatchNorm2d.convert_frozen_batchnorm(self.features)

        out_feature_channels = MobileNetV3_cfg[model_name]["out_feature_channels"]
        self._out_feature_channels = {
            "res2": out_feature_channels[0],
            "res3": out_feature_channels[1],
            "res4": out_feature_channels[2],
            "res5": out_feature_channels[3],
        }
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }

        self.freeze(freeze_at)

    def freeze(self, freeze_at=0):
        # stem + 4 detection stages
        stages = [
            self.features[:1],    # stem
            self.features[1:self.return_features_indices[0]+1],   # up to res2
            self.features[self.return_features_indices[0]+1:self.return_features_indices[1]+1],   # up to res3
            self.features[self.return_features_indices[1]+1:self.return_features_indices[2]+1],  # up to res4
            self.features[self.return_features_indices[2]+1:self.return_features_indices[3]+1], # up to res5
        ]
        for idx, module in enumerate(stages, start=1):
            if freeze_at >= idx:
                for p in module.parameters():
                    p.requires_grad = False

    def forward(self, x):
        outputs = {}

        for i, m in enumerate(self.features):
            x = m(x)
            if i == self.return_features_indices[0] and "res2" in self._out_features:
                outputs["res2"] = x
            elif i == self.return_features_indices[1] and "res3" in self._out_features:
                outputs["res3"] = x
            elif i == self.return_features_indices[2] and "res4" in self._out_features:
                outputs["res4"] = x
            elif i == self.return_features_indices[3] and "res5" in self._out_features:
                outputs["res5"] = x

        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_mobilenet_v3_large_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = MobileNetV3Large(cfg, freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_retina_mobilenet_v3_large_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = MobileNetV3Large(cfg, freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_top = out_channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_top, out_channels, "p5"),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


if __name__ == "__main__":
    x = torch.ones(1, 3, 512, 512)
    model = MobileNetV3Large(None)
    print(model._out_feature_channels)
    outs = model(x)
    for o in outs:
        print(o, outs[o].shape)