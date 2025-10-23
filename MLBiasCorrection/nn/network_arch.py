import numpy as np
import torch
from torch import nn

torch.random.seed()

class Dense(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Dense, self).__init__()
        self.nn_dense=nn.Linear(input_shape,output_shape)
        self.nn_ReLU=nn.ReLU()
    def forward(self,x):
        y=self.nn_ReLU(self.nn_dense(x))
        return y

#Model network defination
class rnn_model(nn.Module):
        
    def __init__(self, parameter_list, name = 'RNN_Model'):
        super(rnn_model, self).__init__()
        self.unit = parameter_list['RNN_output']
        self.acti = parameter_list['activation']
        self.acti_d = parameter_list['d_activation']
        self.recurrent_activ = parameter_list['rec_activation']
#        self.kernel_regular = tf.keras.regularizers.l2(parameter_list['l2_regu'])
#        self.activity_regular = tf.keras.regularizers.l1(parameter_list['l1_regu'])
        self.drop = parameter_list['rnn_dropout']
        self.recurrent_drop = parameter_list['rec_rnn_dropout']
        self.num_layers = parameter_list['num_rnn_layers']
        self.num_dense_layers = parameter_list['num_dense_layers']
        self.dense_out = parameter_list['dense_output']
        self.locality = parameter_list['locality']
        self.type = parameter_list['NN_type']


        self.rnn_list = nn.Sequential()
        i = 0
        for i in range(self.num_layers):
            if i == 0 : 
                input_shape=self.locality
            else :
                input_shape=self.unit[i-1]
            if (self.type == "LSTM") : 
                self.rnn_list.append(nn.LSTMCell(self.locality,self.unit[i]))
            elif (self.type == "GRU") : 
                self.rnn_list.append(nn.GRUCell(self.locality,self.unit[i]))
            elif (self.type == "SimpleRNN") : 
                self.rnn_list.append(nn.RNNCell(self.locality,self.unit[i]))
            elif (self.type == "Dense") : 
                self.rnn_list.append(Dense(self.locality,self.unit[i]))

        self.dense_list = nn.Sequential()
        i = 0
        for i in range(self.num_dense_layers):
            if i == 0 : 
                if self.num_layers == 0:
                   input_shape=self.locality
                else:
                   input_shape=self.unit[-1]
            else :
                input_shape=self.dense_out[i-1]
            self.dense_list.append(Dense(input_shape,self.dense_out[i]))
        self.dense_list.append(nn.Linear(self.dense_out[-1],1))

    def forward(self,x):
        if len(x.shape) < 2: ### not batch
            x=x.unsqueeze(0) 
        batch_size=x.shape[0]
#        for i in range(len(self.rnn_list)):
#            state_h=torch.rand((batch_size,self.unit[i]))
#            state_c=torch.rand((batch_size,self.unit[i]))
#            if (self.type == "LSTM") : 
#              state_h, state_c = self.rnn_list[i](x, (state_h, state_c))
#            elif (self.type == "GRU" or self.type == "SimpleRNN") : 
#              state_h = self.rnn_list[i](x, state_h)
#            elif (self.type == "Dense") :
#              state_h = self.rnn_list[i](x)
#            y = state_h  
#        for i in range(len(self.dense_list)):
#            y = self.dense_list[i](y)
        if len(self.rnn_list) > 0 :
         for i in range(len(self.rnn_list)):
            state_h=torch.rand((batch_size,self.unit[i]),dtype=torch.double)
            state_c=torch.rand((batch_size,self.unit[i]),dtype=torch.double)
            if (self.type == "LSTM") : 
              state_h, state_c = self.rnn_list[i](x, (state_h, state_c))
            elif (self.type == "GRU" or self.type == "SimpleRNN") : 
              state_h = self.rnn_list[i](x, state_h)
            elif (self.type == "Dense") :
              state_h = self.rnn_list[i](x)
            y = state_h  
        else: 
            y = x
        for i in range(len(self.dense_list)):
            y = self.dense_list[i](y) 
        return y
        
