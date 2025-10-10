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

        self.rnn_list.append(nn.LSTMCell(input_size,10))
        self.rnn_list.append(nn.LSTMCell(10,20))

        self.dense_list.append(nn.Linear(20,10))
        self.dense_list.append(nn.Tanh())
        self.dense_list.append(nn.Linear(10,1))
        self.dense_list.append(nn.Tanh())

    def forward(self,x):
        state_h_1=torch.rand(10)
        state_c_1=torch.rand(10)
        state_h_2=torch.rand(20)
        state_c_2=torch.rand(20)
        y, (state_h_1, state_c_1) = self.rnn_list[0](x, (state_h_1, state_c_1) )
        y, (state_h_2, state_c_2) = self.rnn_list[1](y, (state_h_2, state_c_2) )
        y = self.dense_list[1](self.dense_list[0](y))
        y = self.dense_list[3](self.dense_list[2](y))
        return y 

model=rnn_model(5)

torch_input=torch.rand((100,5))
print(model(torch_input).shape)
print(model(torch_input)[0:3])


