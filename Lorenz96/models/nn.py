import os
import numpy as np

import torch
from torch import nn

class nnmodel(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
            nn.Linear(40,40, bias=False),
            nn.ReLU(),
            nn.Linear(40,40, bias=False),
    )
  def forward(self, x):
    y = self.layers(x)
    return y

model=nnmodel()
loss=nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(),lr=1.0e-3)
model_name="nn"

