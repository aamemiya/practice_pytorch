import sys
import os
import numpy as np
import torch 
import network_arch 

plist={"NN_type":"LSTM", "locality":5, "RNN_output":[10,20], "dense_output":[10], "num_rnn_layers":2, "num_dense_layers":1,
"activation":None, "d_activation":None, "rec_activation":None, "l2_regu":None, "l1_regu":None, "rnn_dropout":None,
"rec_rnn_dropout":None}

model=network_arch.rnn_model(plist,5)

torch_input=torch.rand((100,5))
print(model(torch_input).shape)
print(model(torch_input)[0:3])


