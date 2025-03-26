# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import json
import os
from typing import Any, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from PIL.Image import Transpose
from torch.utils.data import Dataset
from torchvision import transforms

from ..boundingbox import utils as bbutils
from ..boundingbox.utils import Height, Width
from PIL import ImageDraw
import math

"""BDD100K object detection dataset module."""


def removesuffix(input_string: str, suffix: str) -> str:
    """Removes suffix string from input string.

    Parameters
    ----------
    input_string : str
        Main input string.
    suffix : str
        Suffix to be removed.

    Returns
    -------
    str
        String without the suffix.
    """
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


class _custom(Dataset):
    def __init__(self,
                 root: str = '.',
                 dataset: str = '.',
                 train: bool = False,
                 seq_len: int = 32,
                 randomize_seq: bool = False,
                 train_images_per_group: int = 100,
                 val_images_per_group: int = 100) -> None:
        """
        Initialize the _BDD dataset.

        Parameters
        ----------
        root : str
            Root directory containing the dataset.
        dataset : str
            Dataset name (e.g., 'track').
        train : bool
            Use training data if True, else validation data.
        seq_len : int
            Number of sequential frames to process at a time.
        randomize_seq : bool
            Randomize sequence start index if True.
        num_train_groups : int
            Number of groups for training data.
        num_val_groups : int
            Number of groups for validation data.
        """
        """
        it is for two json files about train and val.
        """
        super().__init__()
        self.seq_len = seq_len
        self.randomize_seq = randomize_seq

        image_set = 'train' if train else 'val'
        #self.label_file = os.path.join(root, f'labels/box_{dataset}_20/{image_set}/{image_set}.json')
        #self.image_path = os.path.join(root, f'images/{dataset}/{image_set}')
        self.label_file = os.path.join(root, f'labels/{image_set}/{image_set}.json')
        self.image_path = os.path.join(root, f'images/{image_set}')

        try:
            if train_images_per_group < self.seq_len or val_images_per_group < self.seq_len:
                raise ValueError(f"The 'train_images_per_group' or 'val_images_per_group' is smaller than {self.seq_len}. Terminating program.")
        except ValueError as e:
            print(e)
            exit(1)
        
        data_length = len(os.listdir(self.image_path))
        self.num_groups = math.ceil(data_length / train_images_per_group) if train else math.ceil(data_length / val_images_per_group)

        if not os.path.isfile(self.label_file):
            raise FileNotFoundError(f"Label file not found at {self.label_file}")
        if not os.path.isdir(self.image_path):
            raise FileNotFoundError(f"Image folder not found at {self.image_path}")

        # Load and sort data
        with open(self.label_file) as file:
            data = json.load(file)
            # data = data['frames']
        # data.sort(key=lambda img: img['frameIndex'])  # Sort by frameIndex (ascending order)
        data.sort(key=lambda img: img['name']) # Sort by name 

        # Parse categories and assign groups
        categories = set()
        self.groups = {i: [] for i in range(self.num_groups)}  # Initialize groups
        self.annotations = {}
        for idx, img in enumerate(data):
            group_id = idx % self.num_groups  # Distribute evenly into groups
            image_id = img['name']
            self.groups[group_id].append(image_id)
            self.annotations[image_id] = img['labels']
            for cat in img['labels']:
                categories.add(cat['category'])

        self.ids = list(self.groups.keys())  # Group IDs
        print(sorted(list(categories)))
        self.cat_name = sorted(list(categories))  # Sorted category names
        self.idx_map = {name: idx for idx, name in enumerate(self.cat_name)}  # Category to index mapping

    def _get_frame(self, path, labels):
        image = Image.open(path).convert('RGB')
        width, height = image.size
        size = {'height': height, 'width': width}
        objects = []
        for ann in labels:
            name = ann['category']
            bndbox = {'xmin': ann['box2d']['x1'],
                      'ymin': ann['box2d']['y1'],
                      'xmax': ann['box2d']['x2'],
                      'ymax': ann['box2d']['y2']}
            objects.append({'id': self.idx_map[name],
                            'name': name,
                            'bndbox': bndbox})

        annotation = {'annotation': {'size': size, 'object': objects}}
        return image, annotation

    def __getitem__(self, index: int) -> Tuple[torch.tensor, Dict[Any, Any]]:
        group_id = self.ids[index]
        image_ids = self.groups[group_id]

        images = []
        annotations = []
        num_seq = len(image_ids)
        if self.randomize_seq:
            start_idx = np.random.randint(max(num_seq - self.seq_len, 0))
        else:
            start_idx = 0
        stop_idx = start_idx + self.seq_len
        selected_ids = image_ids[start_idx:stop_idx]

        for image_id in selected_ids:
            img_path = os.path.join(self.image_path, image_id)
            image, annotation = self._get_frame(img_path, self.annotations[image_id])
            images.append(image)
            annotations.append(annotation)
            
        # annotation을 각 이미지에 그려서 폴더에 이미지를 저장하는 코드
        # save_path = "/home/mjkim2/Documents/Paper_Experiment/Slayer_sdnn/ground_truth"
        # os.makedirs(save_path, exist_ok=True)

        # for image_id in selected_ids:
        #     img_path = os.path.join(self.image_path, image_id)
        #     image, annotation = self._get_frame(img_path, self.annotations[image_id])

        #     # Draw bounding boxes on the image
        #     draw = ImageDraw.Draw(image)
        #     for obj in annotation['annotation']['object']:
        #         bndbox = obj['bndbox']
        #         draw.rectangle(
        #             [bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax']],
        #             outline="red",
        #             width=2
        #         )
        #         draw.text(
        #             (bndbox['xmin'], bndbox['ymin'] - 10),
        #             obj['name'],
        #             fill="red"
        #         )

        #     # Save the image with bounding boxes
        #     save_img_path = os.path.join(save_path, f"{os.path.splitext(image_id)[0]}_annotated.png")
        #     image.save(save_img_path)

        #     images.append(image)
        #     annotations.append(annotation)

        if len(images) != self.seq_len:
            delta = self.seq_len - len(images)
            images = images + [images[-1]] * delta
            annotations = annotations + [annotations[-1]] * delta

        return images, annotations

    def __len__(self) -> int:
        return len(self.ids)


class custom(Dataset):
    def __init__(self,
                 root: str = './',
                 dataset: str = 'track',
                 size: Tuple[Height, Width] = (448, 448),
                 train: bool = False,
                 seq_len: int = 32,
                 randomize_seq: bool = False,
                 augment_prob: float = 0.0) -> None:
        """Berkley Deep Drive (BDD100K) dataset module. For details on the
        dataset, refer to: https://bdd-data.berkeley.edu/.

        Parameters
        ----------
        root : str, optional
            Root folder where the dataset has been downloaded, by default './'
        dataset : str, optional
            Sub class of BDD100K dataset. By default 'track' which refers to
            MOT2020.
        size : Tuple[Height, Width], optional
            Desired spatial dimension of the frame, by default (448, 448)
        train : bool, optional
            Use training set. If false, testing set is used. By default False.
        seq_len : int, optional
            Number of sequential frames to process at a time, by default 32
        randomize_seq : bool, optional
            Randomize the start of frame sequence. If false, the first seq_len
            of the sample is returned, by default False.
        augment_prob : float, optional
            Augmentation probability of the frames and bounding boxes,
            by default 0.0.
        """
        super().__init__()
        self.blur = transforms.GaussianBlur(kernel_size=5)
        self.color_jitter = transforms.ColorJitter()
        self.grayscale = transforms.Grayscale(num_output_channels=3)
        self.img_transform = transforms.Compose([transforms.Resize(size),
                                                 transforms.ToTensor()])
        self.bb_transform = transforms.Compose([
            lambda x: bbutils.resize_bounding_boxes(x, size),
        ])

        self.datasets = [_custom(root=root, dataset=dataset, train=train,
                              seq_len=seq_len, randomize_seq=randomize_seq)]

        self.classes = self.datasets[0].cat_name
        self.idx_map = self.datasets[0].idx_map
        self.augment_prob = augment_prob
        self.seq_len = seq_len
        self.randomize_seq = randomize_seq

    def flip_lr(self, img) -> Image:
        return Image.Image.transpose(img, Transpose.FLIP_LEFT_RIGHT)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, Dict[Any, Any]]:
        """Get a sample video sequence of BDD100K dataset.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        Tuple[torch.tensor, Dict[Any, Any]]
            Frame sequence and dictionary of bounding box annotations.
        """
        dataset_idx = index // len(self.datasets[0])
        index = index % len(self.datasets[0])
        images, annotations = self.datasets[dataset_idx][index]

        # flip left right
        if np.random.random() < self.augment_prob:
            with ThreadPoolExecutor() as pool:
                images = pool.map(self.flip_lr, images)
            with ThreadPoolExecutor() as pool:
                annotations = pool.map(bbutils.fliplr_bounding_boxes,
                                       annotations)
        # blur
        if np.random.random() < self.augment_prob:
            with ThreadPoolExecutor() as pool:
                images = pool.map(self.blur, images)
        # color jitter
        if np.random.random() < self.augment_prob:
            with ThreadPoolExecutor() as pool:
                images = pool.map(self.color_jitter, images)
        # grayscale
        if np.random.random() < self.augment_prob:
            with ThreadPoolExecutor() as pool:
                images = pool.map(self.grayscale, images)

        with ThreadPoolExecutor() as pool:
            results = pool.map(self.img_transform, images)
            image = torch.stack(list(results), dim=-1)
        annotations = list(map(self.bb_transform, annotations))

        return image, annotations

    def __len__(self) -> int:
        """Number of samples in the dataset.
        """
        return sum([len(dataset) for dataset in self.datasets])
