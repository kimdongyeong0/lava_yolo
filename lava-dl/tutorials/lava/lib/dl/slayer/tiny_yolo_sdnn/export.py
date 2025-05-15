#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Export only the convolutional backbone + detection head of an SDNN YOLO model
to ONNX for clean visualization in Netron.
"""

import os
import torch
import argparse
from lava.lib.dl.slayer import obd


class ConvOnlyExport(torch.nn.Module):
    """Minimal wrapper that runs only the conv blocks + final heads."""
    def __init__(self, orig_model: torch.nn.Module):
        super().__init__()
        # copy references to the modules you actually want to export
        self.blocks = orig_model.blocks      # sigma-delta Conv blocks
        self.heads  = orig_model.heads       # final Conv + Sigma head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add time dim if absent
        if x.dim() == 4:  # [batch, channels, height, width]
            x = x.unsqueeze(-1)
        # run through the conv blocks
        for b in self.blocks:
            x = b(x)
        # run through the head layers
        for h in self.heads:
            x = h(x)
        return x


def export_conv_only_onnx(model_type: str, output_path: str, num_classes: int):
    # 1) instantiate the original SDNN model
    Module = getattr(obd.models, model_type).Network
    orig = Module(
        threshold=0.1,
        tau_grad=0.1,
        scale_grad=0.2,
        num_classes=num_classes,
        clamp_max=5.0
    )
    # prepare internal buffers / anchors
    orig.init_model((448, 448, 3))
    orig.eval()

    # 2) disable all quantize hooks so they don't show up as nodes
    for block in orig.blocks:
        if hasattr(block.synapse, 'pre_hook_fx'):
            block.synapse.pre_hook_fx = lambda t, **kw: t

    # 3) wrap only convs + heads
    minimal = ConvOnlyExport(orig)

    # 4) create dummy input
    dummy = torch.randn(1, 3, 448, 448)

    # 5) export to ONNX
    torch.onnx.export(
        minimal,
        dummy,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        training=torch.onnx.TrainingMode.EVAL,
        verbose=False
    )
    print(f"[✔] conv-only ONNX graph saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export SDNN YOLO conv blocks + head to ONNX"
    )
    parser.add_argument(
        '--model_type', type=str, default='tiny_yolov3_str',
        help='Name of the model class under lava.lib.dl.slayer.obd.models'
    )
    parser.add_argument(
        '--output', type=str, default='conv_only.onnx',
        help='Path to write the ONNX file'
    )
    parser.add_argument(
        '--num_classes', type=int, default=11,
        help='Number of detection classes'
    )
    args = parser.parse_args()
    export_conv_only_onnx(args.model_type, args.output, args.num_classes)
