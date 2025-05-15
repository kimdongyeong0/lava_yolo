import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from typing import List, Tuple, Union
from lava.lib.dl import slayer
from ..yolo_base import YOLOBase

# -----------------------------
# Depthwise Separable Conv w/ Residual
# -----------------------------
class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, neuron_kwargs, in_channels, out_channels,
                 kernel_size, padding=1, stride=1, weight_scale=1, **kwargs):
        super().__init__()
        # Depthwise conv
        self.depthwise = slayer.block.sigma_delta.Conv(
            neuron_kwargs,
            in_channels, in_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            groups=in_channels,
            weight_scale=weight_scale,
            **kwargs
        )
        # Pointwise conv
        self.pointwise = slayer.block.sigma_delta.Conv(
            neuron_kwargs,
            in_channels, out_channels,
            1,
            padding=0,
            stride=1,
            weight_scale=weight_scale,
            **kwargs
        )
        # If dims change or stride >1, project input
        if in_channels != out_channels or stride != 1:
            self.skip_conv = slayer.block.sigma_delta.Conv(
                neuron_kwargs,
                in_channels, out_channels,
                1,
                padding=0,
                stride=stride,
                weight_scale=1,
                **kwargs
            )
        else:
            self.skip_conv = None

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        shortcut = self.skip_conv(x) if self.skip_conv else x
        return F.relu(out + shortcut)

    def export_hdf5(self, handle):
        # implement export if needed
        raise NotImplementedError("Export for depthwise separable conv not implemented")


# -----------------------------
# YOLO-KP Network with Alternating DSConv Blocks
# -----------------------------
class Network(YOLOBase):
    def __init__(self,
                 num_classes: int = 80,
                 anchors: List[List[Tuple[float, float]]] = [
                     [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)],
                 ],
                 threshold: float = 0.1,
                 tau_grad: float = 0.1,
                 scale_grad: float = 0.1,
                 clamp_max: float = 5.0) -> None:
        super().__init__(num_classes=num_classes,
                         anchors=anchors,
                         clamp_max=clamp_max)

        sigma_params = {  # sigma-delta neuron parameters
            'threshold'     : threshold,   # delta unit threshold
            'tau_grad'      : tau_grad,    # delta unit surrogate gradient relaxation parameter
            'scale_grad'    : scale_grad,  # delta unit surrogate gradient scale parameter
            'requires_grad' : True,       # trainable threshold
            'shared_param'  : True,        # layer wise threshold
        }
        sdnn_params = {
            **sigma_params,
            'activation'    : F.relu,      # activation function
        }

        # standard imagenet normalization of RGB images
        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1, 1])
        self.normalize_std  = torch.tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1, 1])

        def quantize_8bit(x: torch.tensor,
                          scale: int = (1 << 6),
                          descale: bool = False) -> torch.tensor:
            return slayer.utils.quantize_hook_fx(x, scale=scale,
                                                 num_bits=8, descale=descale)

        def quantize_5bit(x: torch.tensor,
                          scale: int = (1 << 6),
                          descale: bool = False) -> torch.tensor:
            return slayer.utils.quantize_hook_fx(x, scale=scale,
                                                 num_bits=8, descale=descale)

        synapse_kwargs = dict(weight_norm=False, pre_hook_fx=quantize_8bit)
        block_5_kwargs = dict(weight_norm=True, delay_shift=False, pre_hook_fx=quantize_5bit)
        block_8_kwargs = dict(weight_norm=True, delay_shift=False, pre_hook_fx=quantize_8bit)
        neuron_kwargs = {**sdnn_params, 'norm': slayer.neuron.norm.MeanOnlyBatchNorm}

        self.blocks = torch.nn.ModuleList([
            # Keep early layers as regular convolutions for feature extraction
            slayer.block.sigma_delta.Conv(neuron_kwargs, 3, 16, 3, padding=1, stride=2, weight_scale=1, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 16, 32, 3, padding=1, stride=2, weight_scale=1, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 32, 64, 3, padding=1, stride=2, weight_scale=1, **block_8_kwargs),
            
            # Use depthwise separable convolution to reduce parameters
            DepthwiseSeparableConv(neuron_kwargs, 64, 96, 3, padding=1, stride=2, weight_scale=3, **block_8_kwargs),
            
            # Bottleneck structure (compress-process-expand)
            slayer.block.sigma_delta.Conv(neuron_kwargs, 96, 128, 1, padding=0, stride=1, weight_scale=3, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 128, 128, 3, padding=1, stride=1, weight_scale=3, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 128, 192, 1, padding=0, stride=1, weight_scale=3, **block_8_kwargs),
            
            # Reduced dimensions in later layers
            slayer.block.sigma_delta.Conv(neuron_kwargs, 192, 192, 3, padding=1, stride=2, weight_scale=3, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 192, 384, 3, padding=1, stride=1, weight_scale=3, **block_5_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 384, 192, 1, padding=0, stride=1, weight_scale=3, **block_5_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 192, 384, 3, padding=1, stride=1, weight_scale=3, **block_5_kwargs),
        ])

        self.heads = torch.nn.ModuleList([
            slayer.synapse.Conv(384, self.num_output, 1, padding=0, stride=1, **synapse_kwargs),
            slayer.dendrite.Sigma(),
        ])

    def forward(self, input: torch.Tensor,
                sparsity_monitor: slayer.loss.SparsityEnforcer=None) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
        if self.normalize_mean.device != input.device:
            self.normalize_mean = self.normalize_mean.to(input.device)
            self.normalize_std = self.normalize_std.to(input.device)
        x = (input - self.normalize_mean) / self.normalize_std
        count = []

        for b in self.blocks:
            x = b(x)
            count.append(slayer.utils.event_rate(x))
            if sparsity_monitor:
                sparsity_monitor.append(x)

        for h in self.heads:
            x = h(x)
            count.append(torch.mean((x > 0).to(x.dtype)))

        head1 = self.yolo_raw(x)
        output = [head1] if self.training else self.yolo(head1, self.anchors[0])
        return output, torch.FloatTensor(count).reshape((1, -1)).to(x.device)

    def grad_flow(self, path: str) -> List[torch.tensor]:
        """Montiors gradient flow along the layers.

        Parameters
        ----------
        path : str
            Path for output plot export.

        Returns
        -------
        List[torch.tensor]
            List of gradient norm per layer.
        """
        # helps monitor the gradient flow
        def block_grad_norm(blocks):
            return [b.synapse.grad_norm
                    for b in blocks if hasattr(b, 'synapse')
                    and b.synapse.weight.requires_grad]

        grad = block_grad_norm(self.blocks)

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def load_model(self, model_file: str) -> None:
        pass