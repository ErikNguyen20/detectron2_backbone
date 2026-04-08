#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Descripttion: detectron2 RegNet backbone
# version: 0.0.1
# --------------------------------------------------------

import torch
from torch import nn
from torch.nn import BatchNorm2d

import fvcore.nn.weight_init as weight_init

from detectron2.layers import Conv2d, FrozenBatchNorm2d, ShapeSpec
from detectron2.modeling import Backbone, BACKBONE_REGISTRY, FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool, LastLevelP6P7

from torchvision import models

__all__ = [
    "RegNet",
    "build_regnet_fpn_backbone",
    "build_retina_regnet_fpn_backbone"
]

RegNet_cfg = {
    'regnet_y_800mf': {
        'stages_out_channels': [64, 144, 320, 784],
        'model': models.regnet_y_800mf,
        'weights': models.RegNet_Y_800MF_Weights.DEFAULT,
    },
    'regnet_y_1_6gf': {
        'stages_out_channels': [48, 120, 336, 888],
        'model': models.regnet_y_1_6gf,
        'weights': models.RegNet_Y_1_6GF_Weights.DEFAULT,
    },
}


class RegNet(Backbone):
    """
    Should freeze bn
    """
    def __init__(self, cfg, model_name="regnet_y_800mf", freeze_at=0):
        super(RegNet, self).__init__()
        self._out_features = ["res2", "res3", "res4", "res5"]  # cfg.MODEL.RESNETS.OUT_FEATURES

        if model_name not in RegNet_cfg:
            raise ValueError("Invalid RegNet model")
        else:
            weights = RegNet_cfg[model_name]["weights"]
            model = RegNet_cfg[model_name]["model"](weights=weights)

        self.stem = model.stem
        self.s1 = model.trunk_output.block1
        self.s2 = model.trunk_output.block2
        self.s3 = model.trunk_output.block3
        self.s4 = model.trunk_output.block4

        # Freeze all BN layers
        self.stem = FrozenBatchNorm2d.convert_frozen_batchnorm(self.stem)
        self.s1 = FrozenBatchNorm2d.convert_frozen_batchnorm(self.s1)
        self.s2 = FrozenBatchNorm2d.convert_frozen_batchnorm(self.s2)
        self.s3 = FrozenBatchNorm2d.convert_frozen_batchnorm(self.s3)
        self.s4 = FrozenBatchNorm2d.convert_frozen_batchnorm(self.s4)

        stages_out_channels = RegNet_cfg[model_name]["stages_out_channels"]
        self._out_feature_channels = {
            "res2": stages_out_channels[0],
            "res3": stages_out_channels[1],
            "res4": stages_out_channels[2],
            "res5": stages_out_channels[3],
        }
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }

        self.freeze(freeze_at)

    def freeze(self, freeze_at=0):
        stages = [
            self.stem,
            self.s1,
            self.s2,
            self.s3,
            self.s4,
        ]
        for idx, module in enumerate(stages, start=1):
            if freeze_at >= idx:
                for p in module.parameters():
                    p.requires_grad = False

    def forward(self, x):
        outputs = {}

        x = self.stem(x)

        x = self.s1(x)
        if "res2" in self._out_features:
            outputs["res2"] = x

        x = self.s2(x)
        if "res3" in self._out_features:
            outputs["res3"] = x

        x = self.s3(x)
        if "res4" in self._out_features:
            outputs["res4"] = x

        x = self.s4(x)
        if "res5" in self._out_features:
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
def build_regnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = RegNet(cfg, freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT)
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
def build_retina_regnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = RegNet(cfg, freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT)
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
    model = RegNet(None)
    print(model._out_feature_channels)
    outs = model(x)
    for o in outs:
        print(o, outs[o].shape)