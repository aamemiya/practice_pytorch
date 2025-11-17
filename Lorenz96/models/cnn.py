import os
import numpy as np

import torch
from torch import nn

class nnmodel(nn.Module):
  def __init__(self):
    super().__init__()
    self.in_channel=1
    self.kernel_size=10
    self.out_channel=2
    #self.out_grid=int(40/self.kernel_size) ### from input grid num and kernel size
    self.out_grid=40 ### from input grid num and kernel size
    self.hidden_dim=self.out_channel*self.out_grid
    self.encoder = nn.Sequential(
            nn.Conv1d(self.in_channel,self.out_channel,self.kernel_size,padding="same",padding_mode="circular"), ### ENCODER
            nn.ReLU(),
#            nn.MaxPool1d(self.kernel_size),
#            nn.ReLU(),
            )
    self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), ### PREDICTOR
            nn.ReLU(),
            )
    self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim,40),    ### DECODER
    )
  def forward(self, x):
    y = self.encoder(x.unsqueeze(-2))
#    print(y.shape)
#    quit()
    y = self.predictor(y.flatten(start_dim=-2))
    y = self.decoder(y)
    return y

model=nnmodel()
loss=nn.MSELoss()
optimizer= torch.optim.Adam(model.parameters(),lr=1.0e-2)
model_name="cnn"

x_smp=torch.rand((10,40))
print(x_smp.shape)
print(model(x_smp).shape)

print(model.parameters)

# Total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")

# Trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {trainable_params}")



