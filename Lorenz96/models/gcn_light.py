import os
import numpy as np

import torch
from torch import nn
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.data import Data, Batch
from torch.nn.functional import relu, tanh

class nnmodel(nn.Module):
  def __init__(self, ndim):
    super().__init__()
    self.ndim=ndim
    self.hidden_node=ndim
    self.hidden_feature=5

    encode_stride=5
    encoder_edges=[]
    for j in range(self.hidden_node):
        center_grid = np.mod(int(j),self.ndim)
        for i in range(center_grid-encode_stride,center_grid+encode_stride):
            encoder_edges.append([np.mod(i,self.ndim),j])

    self.encoder_edge_index=torch.tensor(encoder_edges).transpose(1,0)

    self.encoder = GraphConv( (1,self.hidden_feature) , self.hidden_feature)
    self.acti    = relu
    self.final = nn.Linear(self.ndim*self.hidden_feature,self.ndim, bias=False)

  def forward(self, x):
    x=x.unsqueeze(-1)
    nbatch=x.shape[0]
    z=torch.zeros(nbatch,self.hidden_node,self.hidden_feature)

    def batch_index(index_orig,nbatch,node_in,node_out):
        E = index_orig
        row, col = E[0], E[1]
        batched_rows = []
        batched_cols = []
        for b in range(nbatch):
            batched_rows.append(row + b * node_in)   # offset sources
            batched_cols.append(col + b * node_out)  # offset destinations
        index_b = torch.stack([
            torch.cat(batched_rows, dim=0),
            torch.cat(batched_cols, dim=0)
        ], dim=0)
        return index_b

    encoder_edge_index_b=batch_index(self.encoder_edge_index,nbatch,self.ndim,self.hidden_node)

    x_b=torch.cat([x[i] for i in range(nbatch)], dim=0)
    z_b=torch.cat([z[i] for i in range(nbatch)], dim=0)
    z_b = self.acti(self.encoder((x_b,z_b),encoder_edge_index_b))

    z_b = z_b.view(nbatch,self.hidden_node*self.hidden_feature)

    y = self.final(z_b)

    return y

model=nnmodel(40)
loss=nn.MSELoss()
optimizer= torch.optim.Adam(model.parameters(),lr=1.0e-2)

model_name="gcn"

x_smp=torch.rand((10,40))
print(x_smp.shape)
print(model(x_smp).shape)

# Total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")

# Trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {trainable_params}")




