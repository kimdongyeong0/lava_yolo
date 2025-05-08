#!/usr/bin/env python
"""
net2graph.py
Export a trained YOLO-KP network to ONNX format and visualize with Netron.
This version wraps the original model to skip event-rate logic during export,
and patches torch.abs to support Bool tensors.
"""
import warnings
# Suppress common tracing and future warnings during ONNX export
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from torch.jit import TracerWarning
    warnings.filterwarnings("ignore", category=TracerWarning)
except ImportError:
    pass

import argparse
import torch

# Monkey-patch torch.abs to handle Bool tensors
_orig_abs = torch.abs
def _patched_abs(x):
    if x.dtype == torch.bool:
        x = x.to(torch.float)
    return _orig_abs(x)
torch.abs = _patched_abs

# Optional Netron viewer
try:
    import netron
except ImportError:
    netron = None

# Import model registry
from lava.lib.dl.slayer import obd


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export YOLO-KP network to ONNX (skipping event-rate) and optionally launch Netron GUI.')
    parser.add_argument(
        '--model', required=True,
        choices=['tiny_yolov3_str', 'yolo_kp', 'residual_kp', 'residual_str'],
        help='Select network architecture to export')
    parser.add_argument(
        '--weights', type=str, default='',
        help='Path to trained .pt weights file (optional)')
    parser.add_argument(
        '--output', type=str, default='model.onnx',
        help='Output ONNX filename')
    parser.add_argument(
        '--shape', nargs='+', type=int, required=True,
        help='Dummy input shape: e.g. 1 3 32 448 448')
    parser.add_argument(
        '--opset', type=int, default=11,
        help='ONNX opset version')
    parser.add_argument(
        '--no-browser', action='store_true',
        help='Skip launching Netron GUI')
    return parser.parse_args()


class ExportWrapper(torch.nn.Module):
    """
    Wraps the original network to return only detection outputs,
    bypassing event-rate calculation that fails ONNX export.
    """
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        preds, _ = self.net(x)
        # Return only first output tensor if list
        return preds[0] if isinstance(preds, (list, tuple)) else preds


def main():
    args = parse_args()

    # Instantiate network
    ModelClass = getattr(obd.models, args.model).Network
    net = ModelClass().eval()

    # Load pretrained weights if provided
    if args.weights:
        state = torch.load(args.weights, map_location='cpu')
        net.load_state_dict(state)
        print(f'Loaded weights from {args.weights}')

    # Wrap model to skip event-rate logic
    model = ExportWrapper(net)

    # Create dummy input
    dummy = torch.randn(*args.shape)

    # Export to ONNX
    print(f'Exporting {args.model} to ONNX file: {args.output}')
    torch.onnx.export(
        model,
        dummy,
        args.output,
        opset_version=args.opset,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print('ONNX export complete.')

    # Launch Netron if installed
    if not args.no_browser:
        if netron:
            print('Launching Netron viewer...')
            netron.start(args.output)
        else:
            print('Netron not installed; skipping GUI.')


if __name__ == '__main__':
    main()
