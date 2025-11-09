import os
import numpy as np

import torch
from torch import nn

class nnmodel(nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden_dim=10
    self.encoder = nn.Sequential(
            nn.Linear(40,self.hidden_dim), ### ENCODER
            nn.ReLU(),
            )
    self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), ### PREDICTOR
            nn.ReLU(),
            )
    self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim,40),    ### DECODER
            nn.ReLU(),
    )
  def forward(self, x):
    y = self.encoder(x)
    y = self.predictor(y)
    y = self.decoder(y)
    return y

model=nnmodel()
loss=nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(),lr=1.0e-2)
model_name="encoder_decoder"

