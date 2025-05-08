import os
import argparse
from typing import Any, Dict, Tuple
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from lava.lib.dl import slayer
from lava.lib.dl.slayer import obd


# Configuration variables (replacing argparse arguments)
gpu = [0]
threshold = 0.1
tau_grad = 0.1
scale_grad = 0.2
clip = 10

dataset = 'custom'
path = '/home/ysyang/hdd/dataset/udacity'
# path = '/home/bcl/Documents/Sea'
# path = '/home/bcl/Documents/SoccerNet/YOLO'
# path = '/home/bcl/Documents/BDD_dataset/bdd100k_ann'
# path = '/home/bcl/Documents/Sequential_CCTV'
output_dir = '.'
num_workers = 16
aug_prob = 0.2
clamp_max = 5.0
num_classes = 4

# Additional parameters that were referenced but not defined in the argparse version
lr = 1e-3            # Learning rate
wd = 5e-4            # Weight decay
warmup = 5           # Warmup epochs or factor
epoch = 100          # Total number of epochs
lrf = 0.01           # Learning rate final value (as a fraction)
tgt_iou_thr = 0.5    # Target IoU threshold
batch_size = 8      # Batch size

# Set device
device = torch.device('cuda:{}'.format(gpu[0]))

# Define number of classes based on dataset selection
classes_output = {'BDD100K': 11, 'DSIAC': 10, 'custom': num_classes}


# Instantiate the network from the provided module
Network = obd.models.yolo_kp.Network
net = Network(threshold=threshold,
              tau_grad=tau_grad,
              scale_grad=scale_grad,
              num_classes=classes_output[dataset],
              clamp_max=clamp_max).to(device)
module = net

# Initialize the network model with an input shape of (448, 448, 3)
module.init_model((448, 448, 3))

# Set up the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

# Learning rate scheduler function
def lf(x):
    return (min(x / warmup, 1)
            * ((1 + np.cos(x * np.pi / epoch)) / 2)
            * (1 - lrf)
            + lrf)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

# Define YOLO target with the necessary parameters
yolo_target = obd.YOLOtarget(anchors=net.anchors,
                             scales=net.scale,
                             num_classes=net.num_classes,
                             ignore_iou_thres=tgt_iou_thr)

# Load the training and test datasets
train_set = obd.dataset.custom(root=path, dataset='track',
                               train=True, augment_prob=aug_prob,
                               randomize_seq=True)
test_set = obd.dataset.custom(root=path, dataset='track',
                              train=False, randomize_seq=True)

train_set_first = train_set[0]
# Create data loaders for training and testing
train_loader = DataLoader(train_set_first,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=yolo_target.collate_fn,
                          num_workers=num_workers,
                          pin_memory=True)
test_loader = DataLoader(test_set,
                         batch_size=batch_size,
                         shuffle=True,
                         collate_fn=yolo_target.collate_fn,
                         num_workers=num_workers,
                         pin_memory=True)

# Define a color map for drawing boxes (random colors for 11 classes)
box_color_map = [(np.random.randint(256),
                  np.random.randint(256),
                  np.random.randint(256))
                  for i in range(11)]

data_iter = iter(train_loader)
inputs = train_loader.dataset[0]
inputs = inputs.to(device)
print(inputs.shape)

# output_tensor = net.test_layer(inputs, 0)
# print(f"Spike mean: {torch.mean(output_tensor).item():.4f}")

if hasattr(net, 'normalize_mean') and hasattr(net, 'normalize_std'):
    if net.normalize_mean.device != inputs.device:
        net.normalize_mean = net.normalize_mean.to(inputs.device)
        net.normalize_std = net.normalize_std.to(inputs.device)
    inputs_norm = (inputs - net.normalize_mean) / net.normalize_std
else:
    inputs_norm = inputs

# We'll collect per‐layer outputs here for visualization if you like:
visualize_output_tensors = {}

# Loop over each branch:
for branch in ["main", "large", "small"]:
    # Determine how many layers that branch has
    if branch == "main":
        num_layers = len(net.blocks)
    elif branch == "large":
        num_layers = len(net.large_object_branch)
    else:  # small branch
        num_layers = len(net.small_object_branch)

    branch_tensors = []
    print(f"\n=== Branch: {branch} ({num_layers} layers) ===")
    for idx in range(num_layers):
        try:
            # test_layer will do the right preprocessing internally
            out = net.test_layer(inputs_norm, layer_idx=idx, branch=branch)
            mean_spike = out.detach().abs().mean().item()
            print(f"[{branch:5s}][Layer {idx:02d}] shape={tuple(out.shape)}, mean(|spike|)={mean_spike:.4f}")
            branch_tensors.append(out)
        except Exception as e:
            print(f"[{branch:5s}][Layer {idx:02d}] ERROR: {e}")
            break

    visualize_output_tensors[branch] = branch_tensors
    
def visualize_input_frames(tensor, batch_idx=0):
    """
    Visualizes all frames (sequences) within a batch.
    """
    # Ensure tensor is detached and on CPU before processing
    tensor = tensor.detach().cpu()

    # Extract the batch and move frames to first dimension for easy iteration
    frames = tensor[batch_idx]  # Shape: (3, 33, 100, 16)
    frames = frames.permute(3, 0, 1, 2)  # Reshape to (16, 3, 33, 100)

    # Plot each frame
    fig, axes = plt.subplots(4, 8, figsize=(40, 20))  # 2 rows, 8 columns for 16 frames
    axes = axes.flatten()

    for i in range(32):  # Iterate over frames
        img = frames[i].detach().cpu().numpy().transpose(1, 2, 0)  # Move to CPU, detach, convert to (H, W, C)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Frame {i+1}")

    plt.suptitle(f"Batch {batch_idx} - Frames Visualization")
    plt.tight_layout()
    plt.show()

first_main = visualize_output_tensors["main"][0]
visualize_input_frames(first_main)