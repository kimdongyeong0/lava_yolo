# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from lava.lib.dl import slayer

from ..yolo_base import YOLOBase


class Network(YOLOBase):
    """Sigma-Delta YOLO-KP network definition.

    Parameters
    ----------
    num_classes : int, optional
        Number of object classes to predict, by default 80.
    anchors : List[List[Tuple[float, float]]], optional
        Prediction anchor points.
    threshold : float, optional
        Sigma-delta neuron threshold, by default 0.1.
    tau_grad : float, optional
        Surrogate gradient relaxation parameter, by default 0.1.
    scale_grad : float, optional
        Surrogate gradient scale parameter, by default 0.1.
    clamp_max : float, optional
        Maximum clamp value for converting raw prediction to bounding box
        prediction. By default 5.0.
    """
    def __init__(self,
                 num_classes: int = 80,
                 anchors: List[List[Tuple[float, float]]] = [
                     [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)],
                     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
                 ],
                 threshold: float = 0.1,
                 tau_grad: float = 0.1,
                 scale_grad: float = 0.1,
                 clamp_max: float = 5.0) -> None:

        super().__init__(num_classes=num_classes,
                         anchors=anchors,
                         clamp_max=clamp_max)

        sigma_params = {
            'threshold': threshold,
            'tau_grad': tau_grad,
            'scale_grad': scale_grad,
            'requires_grad': True,
            'shared_param': True,
        }
        sdnn_params = {
            **sigma_params,
            'activation': F.relu,
        }

        # standard imagenet normalization of RGB images
        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406])\
                                  .reshape([1, 3, 1, 1, 1])
        self.normalize_std = torch.tensor([0.229, 0.224, 0.225])\
                                 .reshape([1, 3, 1, 1, 1])

        # quantization hooks
        def quantize_8bit(x: torch.Tensor, scale: int = (1 << 6), descale: bool = False):
            return slayer.utils.quantize_hook_fx(x, scale=scale, num_bits=8, descale=descale)

        def quantize_5bit(x: torch.Tensor, scale: int = (1 << 6), descale: bool = False):
            return slayer.utils.quantize_hook_fx(x, scale=scale, num_bits=8, descale=descale)

        synapse_kwargs = dict(weight_norm=False, pre_hook_fx=quantize_8bit)
        block_5_kwargs = dict(weight_norm=True, delay_shift=False, pre_hook_fx=quantize_5bit)
        block_8_kwargs = dict(weight_norm=True, delay_shift=False, pre_hook_fx=quantize_8bit)
        neuron_kwargs = {**sdnn_params, 'norm': slayer.neuron.norm.MeanOnlyBatchNorm}

        # Input processing
        self.input_blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params),
        ])

        # Main feature extraction blocks
        self.blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Conv(neuron_kwargs, 3, 16, 3, padding=1, stride=2, weight_scale=1, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 16, 32, 3, padding=1, stride=2, weight_scale=1, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 32, 64, 3, padding=1, stride=2, weight_scale=1, **block_8_kwargs),
            slayer.block.sigma_delta.Pool(neuron_kwargs, 2, stride=2, weight_scale=1, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 64, 128, 3, padding=1, stride=1, weight_scale=2, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 128, 256, 3, padding=1, stride=1, weight_scale=3, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 256, 256, 3, padding=1, stride=1, weight_scale=3, **block_8_kwargs),
            slayer.block.sigma_delta.Pool(neuron_kwargs, 2, stride=2, weight_scale=1, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 256, 512, 3, padding=1, stride=1, weight_scale=3, **block_5_kwargs),
            slayer.block.sigma_delta.ConvT(neuron_kwargs, 512, 256, 2, stride=2, padding=0, weight_scale=2, **block_5_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 256, 256, 3, padding=1, stride=1, weight_scale=2, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 256, 512, 3, padding=1, stride=1, weight_scale=3, **block_5_kwargs),
        ])

        # Large-object branch
        self.large_object_branch = torch.nn.ModuleList([
            slayer.block.sigma_delta.Conv(neuron_kwargs, 512, 256, 1, padding=0, stride=1, weight_scale=2, **block_5_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 256, 512, 3, padding=1, stride=1, weight_scale=3, **block_5_kwargs),
        ])

        # ── Solution B: Pool 28×28 → 4×4 ──
        self.small_object_branch = torch.nn.ModuleList([
            slayer.block.sigma_delta.Pool(
                neuron_kwargs,
                kernel_size=7,    # 28 ÷ 7 = 4
                stride=7,
                weight_scale=1,
                **block_8_kwargs
            ),
            slayer.block.sigma_delta.Flatten(),
            slayer.block.sigma_delta.Dense(
                neuron_kwargs,
                256 * 4 * 4,      # matches 256×1×4×4
                1024,
                weight_scale=2,
                **block_8_kwargs
            ),
            slayer.block.sigma_delta.Dense(
                neuron_kwargs,
                1024,
                256,
                weight_scale=2,
                **block_8_kwargs
            ),
        ])

        # Detection heads
        self.heads = torch.nn.ModuleList([
            slayer.synapse.Conv(512, self.num_output, 1, padding=0, stride=1, **synapse_kwargs),
            slayer.dendrite.Sigma(),
            slayer.synapse.Dense(256, self.num_output, **synapse_kwargs),
            slayer.dendrite.Sigma(),
        ])

    def forward(
        self,
        input: torch.Tensor,
        sparsity_monitor: slayer.loss.SparsityEnforcer = None
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
        has_sparsity_loss = sparsity_monitor is not None
        if self.normalize_mean.device != input.device:
            self.normalize_mean = self.normalize_mean.to(input.device)
            self.normalize_std = self.normalize_std.to(input.device)
        x = (input - self.normalize_mean) / self.normalize_std

        count = []
        for block in self.input_blocks:
            x = block(x)
            count.append(slayer.utils.event_rate(x))
        intermediate = x

        # Backbone
        features = []
        for i, block in enumerate(self.blocks):
            intermediate = block(intermediate)
            count.append(slayer.utils.event_rate(intermediate))
            if has_sparsity_loss:
                sparsity_monitor.append(intermediate)
            if i == 5:
                features.append(intermediate)

        # Large-object path
        large = intermediate.clone()
        for block in self.large_object_branch:
            large = block(large)
            count.append(slayer.utils.event_rate(large))
            if has_sparsity_loss:
                sparsity_monitor.append(large)

        # Small-object path
        small = features[0]
        for block in self.small_object_branch:
            small = block(small)
            count.append(slayer.utils.event_rate(small))
            if has_sparsity_loss:
                sparsity_monitor.append(small)

        # Heads
        large_head = self.heads[0](large)
        large_head = self.heads[1](large_head)
        small_head = small.reshape(small.shape[0], -1, 1, 1, small.shape[-1])
        small_head = self.heads[2](small_head)
        small_head = self.heads[3](small_head)

        count.append(torch.mean((large_head > 0).float()).item())
        count.append(torch.mean((small_head > 0).float()).item())

        head1 = self.yolo_raw(large_head)
        head2 = self.yolo_raw(small_head)

        if not self.training:
            out1 = self.yolo(head1, self.anchors[0])
            out2 = self.yolo(head2, self.anchors[1])
            output = torch.cat([out1, out2], dim=1)
        else:
            output = [head1, head2]

        return output, torch.FloatTensor(count).reshape((1, -1)).to(input.device)

    # def export_hdf5_head(self, handle: h5py.Dataset) -> None:
    #     def weight(s):
    #         return s.pre_hook_fx(s.weight, descale=True)\
    #                 .reshape(s.weight.shape).cpu().data.numpy()

    #     handle.create_dataset('type', (1,), 'S10', ['conv'.encode('ascii')])
    #     heads_group = handle.create_group('heads')

    #     # Large
    #     lg = heads_group.create_group('large')
    #     syn, neu = self.heads[0], self.heads[1]
    #     lg.create_dataset('shape', data=np.array(neu.shape))
    #     lg.create_dataset('inChannels', data=syn.in_channels)
    #     lg.create_dataset('outChannels', data=syn.out_channels)
    #     lg.create_dataset('kernelSize', data=syn.kernel_size[:-1])
    #     lg.create_dataset('stride', data=syn.stride[:-1])
    #     lg.create_dataset('padding', data=syn.padding[:-1])
    #     lg.create_dataset('dilation', data=syn.dilation[:-1])
    #     lg.create_dataset('groups', data=syn.groups)
    #     lg.create_dataset('weight', data=weight(syn))
    #     dp = neu.device_params; dp['sigma_output'] = True
    #     for k, v in dp.items():
    #         lg.create_dataset(f'neuron/{k}', data=v)

    #     # Small
    #     sm = heads_group.create_group('small')
    #     syn2, neu2 = self.heads[2], self.heads[3]
    #     sm.create_dataset('shape', data=np.array(neu2.shape))
    #     sm.create_dataset('inFeatures', data=syn2.in_features)
    #     sm.create_dataset('outFeatures', data=syn2.out_features)
    #     sm.create_dataset('weight', data=weight(syn2))
    #     dp2 = neu2.device_params; dp2['sigma_output'] = True
    #     for k, v in dp2.items():
    #         sm.create_dataset(f'neuron/{k}', data=v)

    # def export_hdf5(self, filename: str) -> None:
    #     h = h5py.File(filename, 'w')
    #     layer_group = h.create_group('layer')
    #     offset = 0
    #     for i, b in enumerate(self.input_blocks):
    #         b.export_hdf5(layer_group.create_group(f'{i + offset}'))
    #     offset += len(self.input_blocks)
    #     for i, b in enumerate(self.blocks):
    #         b.export_hdf5(layer_group.create_group(f'{i + offset}'))
    #     offset += len(self.blocks)
    #     for i, b in enumerate(self.large_object_branch):
    #         b.export_hdf5(layer_group.create_group(f'{i + offset}'))
    #     offset += len(self.large_object_branch)
    #     for i, b in enumerate(self.small_object_branch):
    #         b.export_hdf5(layer_group.create_group(f'{i + offset}'))
    #     offset += len(self.small_object_branch)
    #     self.export_hdf5_head(layer_group.create_group(f'{offset}'))

    def grad_flow(self, path: str) -> List[torch.Tensor]:
        def block_grad_norm(blocks):
            return [b.synapse.grad_norm
                    for b in blocks
                    if hasattr(b, 'synapse') and b.synapse.weight.requires_grad]

        grad_main = block_grad_norm(self.blocks)
        grad_large = block_grad_norm(self.large_object_branch)
        grad_small = block_grad_norm(self.small_object_branch)
        grad = grad_main + grad_large + grad_small

        plt.figure(figsize=(10, 6))
        plt.semilogy(grad)
        plt.grid(True, which="both", ls="-")
        plt.xlabel("Layers")
        plt.ylabel("Gradient Norm")
        plt.title("Gradient Flow Across Network")
        plt.savefig(path + 'gradFlow.png')
        plt.close()
        return grad

    # def test_layer(self,
    #                input_tensor: torch.Tensor,
    #                layer_idx: int,
    #                branch: str = "main") -> torch.Tensor:
    #     """Test a single layer by index & branch, chaining all preceding layers for correct shape."""
    #     # Normalize input
    #     if self.normalize_mean.device != input_tensor.device:
    #         self.normalize_mean = self.normalize_mean.to(input_tensor.device)
    #         self.normalize_std = self.normalize_std.to(input_tensor.device)
    #     x = (input_tensor - self.normalize_mean) / self.normalize_std

    #     # Pass through input_blocks
    #     for block in self.input_blocks:
    #         x = block(x)

    #     # Branch-specific chaining
    #     if branch == "main":
    #         # Apply main backbone up to layer_idx
    #         for i in range(layer_idx + 1):
    #             x = self.blocks[i](x)
    #         return x

    #     elif branch == "large":
    #         # Run entire backbone first
    #         for block in self.blocks:
    #             x = block(x)
    #         # Then apply large-object branch up to layer_idx
    #         for i in range(layer_idx + 1):
    #             x = self.large_object_branch[i](x)
    #         return x

    #     elif branch == "small":
    #         # Run backbone until feature extraction point (after block 5)
    #         feat = x
    #         for i, block in enumerate(self.blocks):
    #             feat = block(feat)
    #             if i == 5:
    #                 break
    #         # Apply small-object branch up to layer_idx
    #         x_small = feat
    #         for i in range(layer_idx + 1):
    #             x_small = self.small_object_branch[i](x_small)
    #         return x_small

    #     else:
    #         raise ValueError(f"Invalid layer_idx {layer_idx} for branch {branch}")

    def load_model(self, model_file: str) -> None:
        """Selectively load matching parts of a saved state dict."""
        saved_model = torch.load(model_file)
        model_keys = {k: False for k in saved_model.keys()}
        device = self.anchors.device

        # anchors
        if saved_model['anchors'].shape == self.anchors.shape:
            self.anchors.data = saved_model['anchors'].data.to(device)
            model_keys['anchors'] = True
        else:
            print("Warning: anchors shape mismatch. Not loading anchors.")

        # input blocks
        if 'input_blocks.0.neuron.bias' in saved_model:
            self.input_blocks[0].neuron.bias.data = \
                saved_model['input_blocks.0.neuron.bias'].data.to(device)
            model_keys['input_blocks.0.neuron.bias'] = True

        if 'input_blocks.0.neuron.delta.threshold' in saved_model:
            self.input_blocks[0].neuron.delta.threshold.data = \
                saved_model['input_blocks.0.neuron.delta.threshold'].data.to(device)
            model_keys['input_blocks.0.neuron.delta.threshold'] = True

        # helper to load blocks
        def load_block_params(block, prefix, idx):
            key_base = f"{prefix}.{idx}"
            if f"{key_base}.neuron.bias" in saved_model:
                block.neuron.bias.data = saved_model[f"{key_base}.neuron.bias"].data.to(device)
                model_keys[f"{key_base}.neuron.bias"] = True
            if f"{key_base}.neuron.norm.running_mean" in saved_model:
                block.neuron.norm.running_mean.data = \
                    saved_model[f"{key_base}.neuron.norm.running_mean"].data.to(device)
                model_keys[f"{key_base}.neuron.norm.running_mean"] = True
            if f"{key_base}.neuron.delta.threshold" in saved_model:
                block.neuron.delta.threshold.data = \
                    saved_model[f"{key_base}.neuron.delta.threshold"].data.to(device)
                model_keys[f"{key_base}.neuron.delta.threshold"] = True
            if f"{key_base}.synapse.weight_g" in saved_model:
                block.synapse.weight_g.data = \
                    saved_model[f"{key_base}.synapse.weight_g"].data.to(device)
                model_keys[f"{key_base}.synapse.weight_g"] = True
            if f"{key_base}.synapse.weight_v" in saved_model:
                block.synapse.weight_v.data = \
                    saved_model[f"{key_base}.synapse.weight_v"].data.to(device)
                model_keys[f"{key_base}.synapse.weight_v"] = True

        # main blocks
        num_to_load = min(len(self.blocks),
                          sum(k.startswith('blocks.') for k in saved_model) // 5)
        for i in range(num_to_load):
            if hasattr(self.blocks[i], 'neuron') and hasattr(self.blocks[i], 'synapse'):
                load_block_params(self.blocks[i], 'blocks', i)

        # branch blocks
        for prefix, blocks in [('large_object_branch', self.large_object_branch),
                               ('small_object_branch', self.small_object_branch)]:
            for i, b in enumerate(blocks):
                if hasattr(b, 'neuron') and hasattr(b, 'synapse'):
                    load_block_params(b, prefix, i)

        # heads
        if 'heads.0.weight' in saved_model and \
           self.heads[0].weight.data.shape == saved_model['heads.0.weight'].data.shape:
            self.heads[0].weight.data = saved_model['heads.0.weight'].data.to(device)
            model_keys['heads.0.weight'] = True

        if 'heads.2.weight' in saved_model and \
           self.heads[2].weight.data.shape == saved_model['heads.2.weight'].data.shape:
            self.heads[2].weight.data = saved_model['heads.2.weight'].data.to(device)
            model_keys['heads.2.weight'] = True

        # report misses
        missing = [k for k, ok in model_keys.items() if not ok]
        if missing:
            print(f"Info: {len(missing)} parameters were not loaded.")
            if len(missing) < 10:
                for k in missing:
                    print(f"  - {k}")