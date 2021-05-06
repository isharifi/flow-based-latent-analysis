import os
import cv2
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from pseudo_label.train import inspect_dims_transferability


def compute_DVN(G, deformator, main_dir="E:\Sharifirad\Codes\glow-pytorch\celeba_aligned"):
    transform = transforms.Compose([
        transforms.CenterCrop(160),
        transforms.Resize(64),
        transforms.ToTensor()])
    total_imgs = os.listdir(main_dir)
    real_data_train = []
    real_data_val = []
    n = 1000

    for idx in range(n):
        img_loc = os.path.join(main_dir, total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = transform(image)
        real_data_train.append(tensor_image)

    for idx in range(n, 2 * n):
        img_loc = os.path.join(main_dir, total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = transform(image)
        real_data_val.append(tensor_image)

    # Load G and Deformator

    inspect_dims_transferability(G, deformator, 0, real_data_train, real_data_val, r=10, size=None)
