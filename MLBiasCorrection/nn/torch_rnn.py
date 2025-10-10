import numpy as np
import torch
from torch import nn

torch.random.seed()

#Model network defination
class rnn_model(nn.Module):

    def __init__(self, input_size, name = 'RNN_Model'):
        super(rnn_model, self).__init__()
        self.rnn_list = []
        self.dense_list = []

        self.rnn_list.append(nn.RNNCell(input_size,10))

        self.dense_list.append(nn.Linear(10,1))
        self.dense_list.append(nn.Tanh())

    def forward(self,x):
        state_h_1=torch.rand(3,10)
        state_h_1 = self.rnn_list[0](x, state_h_1) 
        y = self.dense_list[1](self.dense_list[0](state_h_1))
        return y 

model=rnn_model(5)

torch_input=torch.rand((3,5))
for i in range(3):
  print(torch_input[i,:])

print(model(torch_input).shape)
for i in range(3):
  print(model(torch_input)[i])


