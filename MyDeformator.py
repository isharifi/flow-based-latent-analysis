import torch
from torch import nn
from torch.nn import functional as F
from enum import Enum
import numpy as np


class MyDeformator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyDeformator, self).__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)
        self.linear.weight.data = 0.1 * torch.randn_like(self.linear.weight.data)

    def forward(self, input):
        output = self.linear(input)
        return output
