import os
import cv2
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from pseudo_label.train import inspect_dims_transferability


def compute_DVN(G, deformator, filename_to_save, main_dir="/data/s2kamyab/sharifirad/glow-pytorch/datasets/anime"):
    transform = transforms.Compose([
        # transforms.CenterCrop(160),
        transforms.Resize((64, 64)),
        transforms.ToTensor()])
    total_imgs = os.listdir(main_dir)
    real_data_train = []
    real_data_val = []
    n = 10000

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
    number_of_directions = 200
    DVN = []
    G = G.to('cuda:0')
    deformator = deformator.to('cuda:0')
    for dim in range(0, number_of_directions):
        DVN.append(inspect_dims_transferability(G, deformator, dim, real_data_train, real_data_val, r=10, size=None))
        print(f'dir {dim}: {DVN[dim][2]}')

    print("Completed")
    torch.save(DVN, filename_to_save)

    d = []
    for i in range(len(DVN)):
        d.append(DVN[i][2])
    dvn = torch.FloatTensor(d)
    print(dvn.mean())

    sorted_dvn = dvn.sort(descending=True)
    print(sorted_dvn[0][:50].mean())

