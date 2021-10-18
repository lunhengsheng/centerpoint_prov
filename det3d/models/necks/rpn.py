import time
import numpy as np
import math

import torch

import torch.nn as nn

from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torch.nn.modules.batchnorm import _BatchNorm

from det3d.torchie.cnn import constant_init, kaiming_init, xavier_init
from det3d.torchie.trainer import load_checkpoint
from det3d.models.utils import Empty, GroupNorm, Sequential
from det3d.models.utils import change_default_args

from .. import builder
from ..registry import NECKS
from ..registry import ENCODERS
from ..registry import DECODERS
from ..utils import build_norm_layer


@NECKS.register_module
class RPN(nn.Module):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        name="rpn",
        logger=None,
        **kwargs
    ):
        super(RPN, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            # print(upsample_strides[i])
            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
            )
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = (self._upsample_strides[i - self._upsample_start_idx])
                if stride > 1:
                    deblock = Sequential(
                        nn.ConvTranspose2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = Sequential(
                        nn.Conv2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self._norm_cfg, planes)[1],
            # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(
                build_norm_layer(self._norm_cfg, planes)[1],
                # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
            block.add(nn.ReLU())

        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, batch_dict):
        ups = []

        x = batch_dict["spatial_features"]

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)

        batch_dict["spatial_features_2d"] = x

        return batch_dict




"""
This is the Encoder in the BaseBackbone2d.
We separate it in order to be able to access the 
encoder features directly.
"""

@ENCODERS.register_module
class BaseBEVEncoder(nn.Module):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        num_filters,
        num_input_channels,
        norm_cfg=None,
        name="BaseBEVEncoder",
        logger=None,
        **kwargs
    ):
        super().__init__()

        self._layer_strides = ds_layer_strides
        self._num_filters = num_filters
        self._layer_nums = layer_nums
        self._input_channels = num_input_channels


        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)

        num_levels = len(self._layer_nums)
        c_in_list = [self._input_channels, *self._num_filters[:-1]]
        self.blocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], self._num_filters[idx], kernel_size=3,
                    stride=self._layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(self._num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(self._layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(self._num_filters[idx], self._num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(self._num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))



    def forward(self, batch_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = batch_dict['spatial_features']
        x = spatial_features

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            batch_dict['spatial_features_%dx' % i] = x
            batch_dict['spatial_features_stride_%dx' % i] = stride
        return batch_dict



"""
We introduce two different BEV decoders - ConcatBEV and ConcatVoxel Decoders.
These decoders, unlike the BaseBEVDecoder concatenates the convolutional
feature map from the encoder, with the self-attention features obtained
after the cfe module operation.
"""

@DECODERS.register_module
class BaseBEVDecoder(nn.Module):
    def __init__(
        self,
        num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_channels,
        norm_cfg=None,
        name="BaseBEVDecoder",
        logger=None,
        **kwargs
    ):
        super().__init__()



        self._upsample_strides = us_layer_strides
        self._num_filters = num_filters
        self._num_upsample_filters = us_num_filters
        self._input_channels = num_input_channels

        self.num_levels = len(num_filters)

        assert len(self._upsample_strides) == len(self._num_upsample_filters)


        self.deblocks = nn.ModuleList()
        for idx in range(self.num_levels):
            if len(self._upsample_strides) > 0:
                stride = self._upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            self._num_filters[idx], self._num_upsample_filters[idx],
                            self._upsample_strides[idx],
                            stride=self._upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(self._num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            self._num_filters[idx], self._num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(self._num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
        c_in = sum(self._num_upsample_filters)
        if len(self._upsample_strides) > self.num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, self._upsample_strides[-1], stride=self._upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        x = spatial_features

        for i in range(self.num_levels):
            stride = int(spatial_features.shape[2] / x.shape[2])
            x = data_dict['spatial_features_%dx' % stride]

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict



@DECODERS.register_module
class ConcatBEVDecoder(BaseBEVDecoder):
    def __init__(
        self,
        in_dim,
        num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_channels,
        norm_cfg=None,
        name="ConcatBEVDecoder",
        logger=None,
        **kwargs
    ):
        super().__init__(num_filters, us_layer_strides, us_num_filters, num_input_channels)


        self._upsample_strides = us_layer_strides
        self._num_filters = num_filters
        self._num_upsample_filters = us_num_filters
        self._input_channels = num_input_channels
        self._in_dim = in_dim


        self.num_levels = len(num_filters)

        assert len(self._upsample_strides) == len(self._num_upsample_filters)

        self.deblocks = nn.ModuleList()
        for idx in range(self.num_levels):
            if len(self._upsample_strides) > 0:
                stride = self._upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            self._in_dim + self._num_filters[idx], self._num_upsample_filters[idx],
                            self._upsample_strides[idx],
                            stride=self._upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(self._num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            self._in_dim + self._num_filters[idx], self._num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(self._num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
        c_in = sum(self._num_upsample_filters)
        if len(self._upsample_strides) > self.num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, self._upsample_strides[-1], stride=self._upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        x = spatial_features

        for i in range(self.num_levels):
            x = data_dict['spatial_features_%dx' % i]
            x = torch.cat([x, data_dict['pillar_context'][i]], dim=1)

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        return data_dict


