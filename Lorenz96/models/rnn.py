import os
import numpy as np

import torch
from torch import nn

class nnmodel(nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn = nn.RNN(40,40)
    self.acti= nn.ReLU()

  def forward(self, x):
    y, _ = self.rnn(x)
    y = self.acti(y)
    return y

model=nnmodel()
loss=nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(),lr=1.0e-2)
model_name="rnn"

