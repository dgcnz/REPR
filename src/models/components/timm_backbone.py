# Source:
# https://github.com/IDEA-Research/detrex/blob/main/detrex/modeling/backbone/timm_backbone.py
#
# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# # Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------------------------------
# Support TIMM Backbone
# Modified from:
# https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/timm_backbone.py
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/backbone.py
# ------------------------------------------------------------------------------------------------

import warnings
from typing import Tuple
import torch.nn as nn

from detectron2.modeling.backbone import Backbone
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

import timm
import torch
from timm.models.vision_transformer import VisionTransformer


def log_timm_feature_info(feature_info):
    """Print feature_info of timm backbone to help development and debug.
    Args:
        feature_info (list[dict] | timm.models.features.FeatureInfo | None):
            feature_info of timm backbone.
    """
    logger = setup_logger(name="timm_backbone")
    if feature_info is None:
        logger.warning("This backbone does not have feature_info")
    elif isinstance(feature_info, list):
        for feat_idx, each_info in enumerate(feature_info):
            logger.info(f"backbone feature_info[{feat_idx}]: {each_info}")
    else:
        try:
            logger.info(f"backbone out_indices: {feature_info.out_indices}")
            logger.info(f"backbone out_channels: {feature_info.channels()}")
            logger.info(f"backbone out_strides: {feature_info.reduction()}")
        except AttributeError:
            logger.warning("Unexpected format of backbone feature_info")


def _load_state_dict_to_features_only(model: VisionTransformer, state_dict: dict, strict: bool = False):
    """Load state dict for features_only model.

    Note:
        - features_only expects keys to start with "model."
        - https://github.com/huggingface/pytorch-image-models/blob/e44f14d7d2f557b9f3add82ee4f1ed2beefbb30d/timm/models/_features.py#L461-L466
        - We may expect the following keys in the state dict to be "unexpected":
            - the `norm` layer (sometimes) is removed from the state dict
            - the `head` layer is removed from the state dict
            - src: https://github.com/huggingface/pytorch-image-models/blob/e44f14d7d2f557b9f3add82ee4f1ed2beefbb30d/timm/models/vision_transformer.py#L790-L805
    """
    logger = setup_logger(name="timm_backbone")
    state_dict = state_dict.copy()
    # add "model." prefix to the keys in state_dict
    state_dict = {f"model.{k}": v for k, v in state_dict.items()}

    miss, unex = model.load_state_dict(state_dict, strict=strict)
    logger.info(f"Missing keys: {miss}")
    logger.info(f"Unexpected keys: {unex}")
    # check missing keys is empty
    assert not miss, f"Missing keys: {miss}"
    # check unexpected keys is only at most "head" and "norm"
    possibly_unexpected_keys = {
        "model.head.weight",
        "model.head.bias",
        "model.norm.weight",
        "model.norm.bias",
    }
    assert set(unex).issubset(possibly_unexpected_keys), f"Unexpected keys: {unex}"


class TimmBackbone(Backbone):
    """A wrapper for using backbone from timm library.
    Please see the document for `feature extraction with timm
    <https://rwightman.github.io/pytorch-image-models/feature_extraction/>`_
    for more details.
    Args:
        model_name (str): Name of timm model to instantiate.
        features_only (bool): Whether to extract feature pyramid (multi-scale
            feature maps from the deepest layer of each stage).
        pretrained (bool): Whether to load pretrained weights. Default: False.
        checkpoint_path (str): Whether to load pretrained weights. Default: False.
        in_channels (int): The number of input channels. Default: 3.
        out_indices (tuple[str]): The extracted feature indices which select
            specific feature levels or limit the stride of the feature extractor.
        out_features (tuple[str]): A map for the output feature dict, e.g.,
            set ("p0", "p1") to return only the feature from indices (0, 1) as
            ``{"p0": feature from indice 0, "p1": feature from indice 1}``.
        norm_layer (nn.Module): Set the specified norm layer for feature extractor,
            e.g., set ``norm_layer=FrozenBatchNorm2d`` to freeze the norm layer
            in feature extractor.
    """

    def __init__(
        self,
        model_name: str,
        features_only: bool = True,
        pretrained: bool = False,
        ckpt_path: str = "",
        pretrained_strict: bool = False,
        in_channels: int = 3,
        out_indices: Tuple[int] = (0, 1, 2, 3),
        norm_layer: nn.Module = None,
        **kwargs,
    ):
        super().__init__()
        logger = setup_logger(name="timm_backbone")
        if not isinstance(pretrained, bool):
            raise TypeError("pretrained must be bool, not str for model path")

        try:
            self.timm_model: nn.Module = timm.create_model(
                model_name=model_name,
                features_only=features_only,
                pretrained=pretrained,
                in_chans=in_channels,
                out_indices=out_indices,
                norm_layer=norm_layer,
                **kwargs,
            )
            if ckpt_path:
                logger.info(f"Loading checkpoint from {ckpt_path}")
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

                if features_only:
                    _load_state_dict_to_features_only(
                        self.timm_model, state_dict
                    )
                else:
                    miss, unex = self.timm_model.load_state_dict(
                        state_dict, strict=pretrained_strict
                    )
                    logger.info(f"Loaded checkpoint from {ckpt_path}")
                    logger.info(f"Missing keys: {miss}")
                    logger.info(f"Unexpected keys: {unex}")
                    assert not miss, f"Missing keys: {miss}"
                    assert len(unex) < 4, f"Unexpected keys: {unex}"
                
                logger.info(f"Loaded checkpoint from {ckpt_path}")
        except Exception as error:
            if "feature_info" in str(error):
                raise AttributeError(
                    "Using features_only may cause attribute error"
                    " in timm, cause there's no feature_info attribute in some models. See "
                    "https://github.com/rwightman/pytorch-image-models/issues/1438"
                )
            elif "norm_layer" in str(error):
                raise ValueError(
                    f"{model_name} does not support specified norm layer, please set 'norm_layer=None'"
                )
            else:
                logger.info(error)
                exit()

        self.out_indices = out_indices

        feature_info = getattr(self.timm_model, "feature_info", None)
        if comm.get_rank() == 0:
            log_timm_feature_info(feature_info)

        if feature_info is not None:
            output_feature_channels = {
                "p{}".format(out_indices[i]): feature_info.channels()[i]
                for i in range(len(out_indices))
            }
            out_feature_strides = {
                "p{}".format(out_indices[i]): feature_info.reduction()[i]
                for i in range(len(out_indices))
            }

            self._out_features = {
                "p{}".format(out_indices[i]) for i in range(len(out_indices))
            }
            self._out_feature_channels = {
                feat: output_feature_channels[feat] for feat in self._out_features
            }
            self._out_feature_strides = {
                feat: out_feature_strides[feat] for feat in self._out_features
            }

    def forward(self, x):
        """Forward function of `TimmBackbone`.
        Args:
            x (torch.Tensor): the input tensor for feature extraction.
        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "p1") to tensor
        """
        features = self.timm_model(x)
        outs = {}
        for i in range(len(self.out_indices)):
            out = features[i]
            outs["p{}".format(self.out_indices[i])] = out

        return outs
