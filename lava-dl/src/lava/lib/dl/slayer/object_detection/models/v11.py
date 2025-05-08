# SPDX-License-Identifier: BSD-3-Clause
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from lava.lib.dl import slayer
from ..yolo_base import YOLOBase


class SpatialPool(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W, T]
        B, C, H, W, T = x.shape
        y = x.permute(0, 4, 1, 2, 3).reshape(B * T, C, H, W)
        y = F.max_pool2d(y, kernel_size=self.kernel_size,
                         stride=self.stride, padding=self.padding)
        H2, W2 = y.shape[2], y.shape[3]
        y = y.reshape(B, T, C, H2, W2).permute(0, 2, 3, 4, 1)
        return y


class C3k2(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 sdnn_params: dict,
                 block_kwargs: dict,
                 synapse_kwargs: dict) -> None:
        super().__init__()
        self.conv1 = slayer.block.sigma_delta.Conv(
            sdnn_params, in_ch, in_ch, kernel_size=2,
            padding=0, stride=1, **block_kwargs
        )
        self.conv2 = slayer.block.sigma_delta.Conv(
            sdnn_params, in_ch, in_ch, kernel_size=2,
            padding=0, stride=1, **block_kwargs
        )
        self.project = slayer.synapse.Conv(
            in_ch * 2, out_ch, 1, padding=0, stride=1,
            **synapse_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        return self.project(torch.cat([y1, y2], dim=1))


class Network(YOLOBase):
    def __init__(self,
                 num_classes: int = 80,
                 anchors: List[List[Tuple[float, float]]] = [
                     [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)],
                     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
                 ],
                 threshold: float = 0.1,
                 tau_grad: float    = 0.1,
                 scale_grad: float  = 0.1,
                 clamp_max: float   = 5.0) -> None:
        super().__init__(num_classes=num_classes,
                         anchors=anchors,
                         clamp_max=clamp_max)

        sigma_params = {
            'threshold':     threshold,
            'tau_grad':      tau_grad,
            'scale_grad':    scale_grad,
            'requires_grad': False,
            'shared_param':  True,
        }
        sdnn_params  = {**sigma_params, 'activation': F.relu}

        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406])\
                                  .reshape(1, 3, 1, 1, 1)
        self.normalize_std  = torch.tensor([0.229, 0.224, 0.225])\
                                  .reshape(1, 3, 1, 1, 1)

        def _quantize_8bit(x: torch.Tensor, scale=(1 << 6), descale=False):
            return slayer.utils.quantize_hook_fx(
                x, scale=scale, num_bits=8, descale=descale
            )

        synapse_kwargs = dict(weight_norm=False,
                              pre_hook_fx=_quantize_8bit)
        block_kwargs   = dict(weight_norm=True,
                              delay_shift=False,
                              pre_hook_fx=_quantize_8bit)
        neuron_kwargs  = {**sdnn_params,
                          'norm': slayer.neuron.norm.MeanOnlyBatchNorm}

        self.input_blocks = nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params),
        ])

        self.backend_blocks = nn.ModuleList([
            slayer.block.sigma_delta.Conv(neuron_kwargs,  3,  16, 3,
                                          padding=1, stride=2,
                                          weight_scale=1, **block_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 16,  32, 3,
                                          padding=1, stride=2,
                                          weight_scale=1, **block_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 32,  64, 3,
                                          padding=1, stride=2,
                                          weight_scale=1, **block_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 64, 128, 3,
                                          padding=1, stride=2,
                                          weight_scale=3, **block_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs,128, 256, 3,
                                          padding=1, stride=1,
                                          weight_scale=3, **block_kwargs),
        ])

        self.sppf_pools = nn.ModuleList([
            SpatialPool(5, 1, 2),
            SpatialPool(9, 1, 4),
            SpatialPool(13,1, 6),
        ])
        self.sppf_proj     = slayer.synapse.Conv(4*256, 256, 1,
                                                 padding=0, stride=1,
                                                 **synapse_kwargs)
        self.c3k2_backbone = C3k2(256, 256,
                                  sdnn_params, block_kwargs,
                                  synapse_kwargs)

        self.head1_backend = nn.ModuleList([
            slayer.block.sigma_delta.Conv(neuron_kwargs,256,256,3,
                                          padding=1, stride=2,
                                          **block_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs,256,512,3,
                                          padding=1, stride=1,
                                          **block_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs,512,1024,3,
                                          padding=1, stride=1,
                                          **block_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs,1024,256,1,
                                          padding=0, stride=1,
                                          **block_kwargs),
        ])
        self.head1_blocks = nn.ModuleList([
            slayer.block.sigma_delta.Conv(neuron_kwargs,256,512,3,
                                          padding=1, stride=1,
                                          **block_kwargs),
            slayer.synapse.Conv(512, self.num_output, 1,
                                padding=0, stride=1,
                                **synapse_kwargs),
            slayer.dendrite.Sigma(),
        ])

        self.head2_backend = nn.ModuleList([
            slayer.block.sigma_delta.Conv(neuron_kwargs,256,128,1,
                                          padding=0, stride=1,
                                          **block_kwargs),
            slayer.block.sigma_delta.Unpool(sdnn_params,
                                            kernel_size=2,
                                            stride=2,
                                            **block_kwargs),
        ])
        self.c3k2_fpn     = C3k2(128+256,384,
                                 sdnn_params, block_kwargs,
                                 synapse_kwargs)
        self.head2_blocks = nn.ModuleList([
            slayer.block.sigma_delta.Conv(neuron_kwargs,384,256,3,
                                          padding=1, stride=1,
                                          **block_kwargs),
            slayer.synapse.Conv(256, self.num_output, 1,
                                padding=0, stride=1,
                                **synapse_kwargs),
            slayer.dendrite.Sigma(),
        ])


    def forward(self,
                input: torch.Tensor,
                sparsity_monitor: slayer.loss.SparsityEnforcer = None
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
        if self.normalize_mean.device != input.device:
            self.normalize_mean = self.normalize_mean.to(input.device)
            self.normalize_std  = self.normalize_std.to(input.device)
        x = (input - self.normalize_mean) / self.normalize_std

        count = []
        for blk in self.input_blocks:
            x = blk(x)
            count.append(slayer.utils.event_rate(x))

        for blk in self.backend_blocks:
            x = blk(x)
            count.append(slayer.utils.event_rate(x))
        backbone_feat = x

        pools = [p(backbone_feat) for p in self.sppf_pools]
        for p in pools:
            count.append(slayer.utils.event_rate(p))
        x = torch.cat([backbone_feat, *pools], dim=1)
        count.append(slayer.utils.event_rate(x))
        x = self.sppf_proj(x)
        count.append(slayer.utils.event_rate(x))

        x = self.c3k2_backbone(x)
        count.append(slayer.utils.event_rate(x))

        # HEAD1: capture first result for head2
        h1 = x
        for idx, blk in enumerate(self.head1_backend):
            h1 = blk(h1)
            count.append(slayer.utils.event_rate(h1))
            if idx == 0:
                h2_base = h1.clone()

        for blk in self.head1_blocks:
            h1 = blk(h1)
            count.append(slayer.utils.event_rate(h1))

        # HEAD2: start from that first head1 feature
        h2 = h2_base
        for blk in self.head2_backend:
            h2 = blk(h2)
            count.append(slayer.utils.event_rate(h2))

        fused = torch.cat([h2, backbone_feat], dim=1)
        count.append(slayer.utils.event_rate(fused))
        fused = self.c3k2_fpn(fused)
        count.append(slayer.utils.event_rate(fused))
        for blk in self.head2_blocks:
            fused = blk(fused)
            count.append(slayer.utils.event_rate(fused))

        raw1 = self.yolo_raw(h1)
        raw2 = self.yolo_raw(fused)

        if self.training:
            out = [raw1, raw2]
        else:
            out = torch.cat([
                self.yolo(raw1, self.anchors[0]),
                self.yolo(raw2, self.anchors[1])
            ], dim=1)

        stats = torch.FloatTensor(count).reshape((1, -1)).to(x.device)
        return out, stats


    def grad_flow(self, path: str) -> List[torch.Tensor]:
        def block_grad_norm(blocks):
            return [
                b.synapse.grad_norm
                for b in blocks
                if hasattr(b, 'synapse') and b.synapse.weight.requires_grad
            ]

        grad = []
        # Process module lists
        for block_list in [
            self.input_blocks,
            self.backend_blocks,
            self.head1_backend,
            self.head1_blocks,
            self.head2_backend,
            self.head2_blocks
        ]:
            grad += block_grad_norm(block_list)
        # Process individual modules by wrapping them in a list
        for module in [self.sppf_proj, self.c3k2_backbone, self.c3k2_fpn]:
            grad += block_grad_norm([module])

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()
        return grad
