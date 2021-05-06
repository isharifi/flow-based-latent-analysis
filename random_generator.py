import numpy as np
import torch


class random_generator:
    def __init__(self, dim_size):
        self.Iman_size = dim_size
        self.dim_shift = dim_size * dim_size * 3
        self.dim_z = dim_size * dim_size * 3

    def decode(self, z):
        return torch.rand([2, 3, self.Iman_size, self.Iman_size]).cuda()
