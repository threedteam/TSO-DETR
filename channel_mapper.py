# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmdet.models.necks.scconv import ScConv, SRU

from ..builder import NECKS

class h_sigmoid(nn.Module):
    """Applies the hard sigmoid function element-wise.

    The hard sigmoid function is defined as:
    relu(x + 3) / 6

    Args:
        inplace (bool, optional): If set to True, will modify the input tensor in-place. Default: True.
    """

    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        """Applies the hard sigmoid function element-wise."""
        return self.relu(x + 3) / 6

class PLVPooling(nn.Module):
    def __init__(self):
        super(PLVPooling, self).__init__()

    def forward(self, x, bias):
        batch, C, H, W = x.shape
        out = torch.mean(
            torch.greater(x, bias.view(C,1,1).expand((C, H, W))).float(),
            dim=[2, 3],
        ).view(batch,C,1,1)
        return out

class MixECALayer(nn.Module):
    """Effective Channel Attention."""

    def __init__(self, k_size=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.plv_pool = PLVPooling()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = h_sigmoid()

    def forward(self, x, bias):
        """ECA layer is very close to MLP but with a representation to 1d form."""
        merged = self.avg_pool(x)+self.plv_pool(x,bias)

        # w/o plv
        # merged = self.avg_pool(x)

        y = self.conv(merged.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ConvModuleWrapper(nn.Module):
    def __init__(self,
                in_channel,
                out_channels,
                kernel_size,
                stride: int = 1,
                padding: int = 0,
                conv_cfg: Any = None,
                norm_cfg: Any = None,
                act_cfg: Any = dict(type='ReLU'),
        ) -> None:
        super().__init__()
        self.conv_module=ConvModule(
                    in_channel,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=True)
        self.sru = SRU(out_channels)
        self.channel_attn=MixECALayer(5)

    def forward(self,x):
        x = self.conv_module(x)
        # # original conv_module
        # return x # 2024年1月11日20:55:17 cascade系列模型不需要使用。
        # w/ sru
        x = self.sru(x)
        # w/ mixca
        return self.channel_attn(x, self.conv_module.conv.bias)
    
@NECKS.register_module()
class ChannelMapper(BaseModule):
    r"""Channel Mapper to reduce/increase channels of backbone features.

    This is used to reduce/increase channels of backbone features.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): kernel_size for reducing channels (used
            at each scale). Default: 3.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU').
        num_outs (int, optional): Number of output feature maps. There
            would be extra_convs when num_outs larger than the length
            of in_channels.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = ChannelMapper(in_channels, 11, 3).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 num_outs=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(ChannelMapper, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.extra_convs = None
        if num_outs is None:
            num_outs = len(in_channels)
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                ConvModuleWrapper(
                    in_channel,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels), num_outs):
                if i == len(in_channels):
                    in_channel = in_channels[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(
                    ConvModuleWrapper(
                        in_channel,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[-1]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        return tuple(outs)
