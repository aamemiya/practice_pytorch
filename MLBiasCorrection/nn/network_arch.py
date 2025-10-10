import numpy as np
import torch
from torch import nn

torch.random.seed()

#Model network defination
class rnn_model(nn.Module):

    def __init__(self, parameter_list, name = 'RNN_Model'):
        super(rnn_model, self).__init__()
        self.unit = parameter_list['RNN_output']
        self.acti = parameter_list['activation']
        self.acti_d = parameter_list['d_activation']
        self.recurrent_activ = parameter_list['rec_activation']
        self.kernel_regular = tf.keras.regularizers.l2(parameter_list['l2_regu'])
        self.activity_regular = tf.keras.regularizers.l1(parameter_list['l1_regu'])
        self.drop = parameter_list['rnn_dropout']
        self.recurrent_drop = parameter_list['rec_rnn_dropout']
        self.num_layers = parameter_list['num_rnn_layers']
        self.num_dense_layers = parameter_list['num_dense_layers']
        self.dense_out = parameter_list['dense_output']
        self.locality = parameter_list['locality']
        self.type = parameter_list['NN_type']

        if self.acti_d == "ReLU": 
          self.dense_acti = nn.ReLU
        elif self.acti_d == "tanh": 
          self.dense_acti = nn.Tanh
        else
          print("self.acti_d = " + self.acti_d + " not supported ")
          quit()

        self.rnn_list = []
        i = 0
        for i in range(self.num_layers):
            if (self.type == "LSTM") : 
                self.rnn_list.append(nn.LSTM(input_size=self.locality, 
                                             output_size=self.unit[i],
                                                    activation = self.acti,
                                                    recurrent_activation = self.recurrent_activ,
                                                    kernel_regularizer = self.kernel_regular,
                                                    activity_regularizer = self.activity_regular,
                                                    dropout = self.drop,
                                                    recurrent_dropout = self.recurrent_drop,
                                                    return_sequences = True,
                                                    name = 'LSTM_{}'.format(i+1), 
                                                    return_state=True))
            elif (self.type == "GRU") : 
                self.rnn_list.append(tf.keras.layers.GRU(units = self.unit[i], 
                                                    activation = self.acti,
                                                    recurrent_activation = self.recurrent_activ,
                                                    kernel_regularizer = self.kernel_regular,
                                                    activity_regularizer = self.activity_regular,
                                                    dropout = self.drop,
                                                    recurrent_dropout = self.recurrent_drop,
                                                    return_sequences = True,
                                                    name = 'GRU_{}'.format(i+1), 
                                                    return_state=True))

            elif (self.type == "SimpleRNN") : 
               self.rnn_list.append(tf.keras.layers.SimpleRNN(units = self.unit[i], 
                                                    activation = self.acti,
#                                                    recurrent_activation = self.recurrent_activ,
                                                    kernel_regularizer = self.kernel_regular,
                                                    activity_regularizer = self.activity_regular,
                                                    dropout = self.drop,
                                                    recurrent_dropout = self.recurrent_drop,
                                                    return_sequences = True,
                                                    name = 'RNN_{}'.format(i+1), 
                                                    return_state=True))

            elif (self.type == "Dense") : 
               self.rnn_list.append(tf.keras.layers.Dense(units = self.unit[i], 
                                                    activation = self.acti,
#                                                    recurrent_activation = self.recurrent_activ,
                                                    kernel_regularizer = self.kernel_regular,
                                                    activity_regularizer = self.activity_regular,
                                                    name = 'RNN_{}'.format(i+1), 
                                                    ))



        self.dense_list = []
        i = 0
        for i in range(self.num_dense_layers):
            self.dense_list.append(tf.keras.layers.Dense(units=self.dense_out[i],
                                    kernel_regularizer = self.kernel_regular,
                                    activation = self.acti_d,
                                    name = 'DENSE_{}'.format(i+1)))
        self.dense_list.append(tf.keras.layers.Dense(units=self.dense_out[-1],
                                            kernel_regularizer = self.kernel_regular,
                                            activation = None,
                                            name = 'DENSE_{}'.format(i+1)))
    def forward:



      for i in range(len(self.dense_list)):
        y = self.dense_list[i](y)
      return y

    def call(self, inputs, stat):
        
        initializer = tf.initializers.GlorotUniform(seed = 1)
        state_h = [initializer([inputs.shape[0], self.unit[i]], dtype = tf.dtypes.float32) for i in range(self.num_layers)]
        state_c = [initializer([inputs.shape[0], self.unit[i]], dtype = tf.dtypes.float32) for i in range(self.num_layers)]
        x = inputs
        for i in range(len(self.rnn_list)):
            if (self.type == "LSTM") : 
              try:
                  x, state_h[i], state_c[i] = self.rnn_list[i](x, initial_state = [stat[0][i], stat[1][i]])
              except:
                  x, state_h[i], state_c[i] = self.rnn_list[i](x, initial_state = [state_h[i], state_c[i]])
            if (self.type == "GRU" or self.type == "SimpleRNN") : 
              try:
                  x, state_h[i] = self.rnn_list[i](x, initial_state = [stat[0][i]])
              except:
                  x, state_h[i] = self.rnn_list[i](x, initial_state = [state_h[i]])
            if (self.type == "Dense") : 
                x = self.rnn_list[i](x)
        
        #Only using last time-step as the input to the dense layer
        try:
            x = x[:, -1, :]
        except:
            pass

        for i in range(len(self.dense_list)):
            y = self.dense_acti(self.dense_list[i](y))

        y = layer_output(y)
        return y
        
