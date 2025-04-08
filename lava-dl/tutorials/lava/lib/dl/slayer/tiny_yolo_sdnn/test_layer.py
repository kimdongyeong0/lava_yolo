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
from torch.utils.tensorboard import SummaryWriter

from lava.lib.dl import slayer
from lava.lib.dl.slayer import obd
from torchinfo import summary

import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=int, default=[0], help='which gpu(s) to use', nargs='+')
parser.add_argument('-threshold',  type=float, default=0.1, help='neuron threshold')
parser.add_argument('-tau_grad',   type=float, default=0.1, help='surrogate gradient time constant')
parser.add_argument('-scale_grad', type=float, default=0.2, help='surrogate gradient scale')
parser.add_argument('-clip',       type=float, default=10, help='gradient clipping limit')

parser.add_argument('-dataset',     type=str,   default='custom')
parser.add_argument('-path',        type=str,   default='/home/bcl/Documents/udacity', help='dataset path to use ["data/prophesee", "data/bdd100k", "data/dsiac"]')
parser.add_argument('-output_dir',  type=str,   default='.', help='directory in which to put log folders')
parser.add_argument('-num_workers', type=int,   default=16, help='number of dataloader workers')
parser.add_argument('-aug_prob',    type=float, default=0.2, help='training augmentation probability')
parser.add_argument('-clamp_max',   type=float, default=5.0, help='exponential clamp in height/width calculation')

parser.add_argument('-num_classes', type=int, default=4, help='number of classes')

args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.gpu[0]))

classes_output = {'BDD100K': 11, 'DSIAC': 10, 'custom': args.num_classes}

Network = obd.models.yolo_kp.Network

net = Network(threshold=args.threshold,
                      tau_grad=args.tau_grad,
                      scale_grad=args.scale_grad,
                      num_classes=classes_output[args.dataset],
                      clamp_max=args.clamp_max).to(device)
module = net
        
module.init_model((448, 448, 3))

optimizer = torch.optim.Adam(net.parameters(),
                               lr=args.lr,
                                weight_decay=args.wd)

def lf(x):
        return (min(x / args.warmup, 1)
                * ((1 + np.cos(x * np.pi / args.epoch)) / 2)
                * (1 - args.lrf)
                + args.lrf)
        
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
yolo_target = obd.YOLOtarget(anchors=net.anchors,
                                 scales=net.scale,
                                 num_classes=net.num_classes,
                                 ignore_iou_thres=args.tgt_iou_thr)

train_set = obd.dataset.custom(root=args.path, dataset='track',
                                train=True, augment_prob=args.aug_prob,
                                randomize_seq=True)
test_set = obd.dataset.custom(root=args.path, dataset='track',
                                train=False, randomize_seq=True)
train_set_first = train_set[0]
train_loader = DataLoader(train_set_first,
                            batch_size=args.b,
                            shuffle=True,
                            collate_fn=yolo_target.collate_fn,
                            num_workers=args.num_workers,
                            pin_memory=True)
test_loader = DataLoader(test_set,
                            batch_size=args.b,
                            shuffle=True,
                            collate_fn=yolo_target.collate_fn,
                            num_workers=args.num_workers,
                            pin_memory=True)

box_color_map = [(np.random.randint(256),
                    np.random.randint(256),
                    np.random.randint(256))
                    for i in range(11)]

data_iter = iter(train_loader)
inputs = train_loader.dataset[0]
inputs = inputs.to(device)
# print(inputs.shape)

# output_tensor = net.test_layer(inputs, 0)
# print(f"Spike mean: {torch.mean(output_tensor).item():.4f}")
visualize_output_tensors = []
visualize_output_tensors.append(inputs)

for i in range(12):
    output_tensor = net.test_layer(inputs, i)
    print(output_tensor.shape)
    visualize_output_tensors.append(output_tensor)
    print(f"Spike mean: {torch.mean(output_tensor).item():.4f}")
    inputs = output_tensor
    
    
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
    fig, axes = plt.subplots(4, 4, figsize=(30, 10))  # 2 rows, 8 columns for 16 frames
    axes = axes.flatten()

    for i in range(16):  # Iterate over frames
        img = frames[i].detach().cpu().numpy().transpose(1, 2, 0)  # Move to CPU, detach, convert to (H, W, C)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Frame {i+1}")

    plt.suptitle(f"Batch {batch_idx} - Frames Visualization")
    plt.tight_layout()
    plt.savefig("visualize_input_frames.png")
    plt.close()

First_Layer_Tensor = visualize_output_tensors[0]
visualize_input_frames(First_Layer_Tensor)


def visualize_spike_frames(tensor, batch_idx=0):
    """
    Visualizes all frames (sequences) within a batch.
    """
    # Ensure tensor is detached and on CPU before processing
    tensor = tensor.detach().cpu()

    # Extract the batch and move frames to first dimension for easy iteration
    frames = tensor[batch_idx]  # Shape: (3, 33, 100, 16)
    frames = frames.permute(3, 0, 1, 2)  # Reshape to (16, 3, 33, 100)

    # Plot each frame
    fig, axes = plt.subplots(4, 4, figsize=(30, 10))  # 2 rows, 8 columns for 16 frames
    axes = axes.flatten()

    for i in range(16):  # Iterate over frames
        img = frames[i].detach().cpu().numpy().transpose(1, 2, 0)  # Move to CPU, detach, convert to (H, W, C)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Frame {i+1}")

    plt.suptitle(f"Batch {batch_idx} - Frames Visualization")
    plt.tight_layout()
    plt.savefig("visualize_spike_frames.png")
    plt.close()

First_Layer_Tensor = visualize_output_tensors[1]
visualize_spike_frames(First_Layer_Tensor)

def visualize_tensor_images(tensor, batch_idx, channel_idx):
    """
    Visualizes 16 frames from a specific batch and channel.
    """
    # frames = tensor[batch_idx, channel_idx]  # Shape: (16, 49, 16)
    
    fig, axes = plt.subplots(channel_idx*2, 8, figsize=(50, 20))  # 2 rows, 8 columns
    axes = axes.flatten()

    for j in range(channel_idx):
        frames = tensor[batch_idx, j]
        for i in range(16):  # Loop over frames
            img = frames[:,:,i].detach().cpu().numpy()  # Move to CPU and convert to NumPy
            axes[i+(16*j)].imshow(img)  # Use grayscale if single-channel
            axes[i+(16*j)].axis('off')
            axes[i+(16*j)].set_title(f"C:{j+1}|F:{i+1}")

    # plt.suptitle(f"First Conv Layer Visualization")
    plt.tight_layout()
    plt.savefig("visualize_tensor_images.png")
    plt.close()

print(visualize_output_tensors[2].shape)
First_Layer_Tensor = visualize_output_tensors[2]
visualize_tensor_images(First_Layer_Tensor, 0, 4) ## Total Channel = 24

print(visualize_output_tensors[3].shape)
First_Layer_Tensor = visualize_output_tensors[3]
visualize_tensor_images(First_Layer_Tensor, 0, 4) ## Total Channel = 36

print(visualize_output_tensors[4].shape)
First_Layer_Tensor = visualize_output_tensors[4]
visualize_tensor_images(First_Layer_Tensor, 0, 4) ## Total Channel = 48

print(visualize_output_tensors[5].shape)
First_Layer_Tensor = visualize_output_tensors[5]
visualize_tensor_images(First_Layer_Tensor, 0, 4) ## Total Channel = 64