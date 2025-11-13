import os
import numpy as np

import torch
from torch import nn
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.data import Data, Batch
from torch.nn.functional import relu

class nnmodel(nn.Module):
  def __init__(self, ndim):
    super().__init__()
    self.ndim=ndim
    self.hidden_node=10
    self.hidden_feature=8
    self.hidden_dim=self.hidden_node*self.hidden_feature

    stride=3
    edges=[]
    edge_weights=[]
    for j in range(self.hidden_node):
      edges.append([j,j])
      edge_weights.append(1)
      for dj in range(1,stride-1):
        edges.append([j,np.mod(j-dj,self.hidden_node)])
        edge_weights.append(np.exp( -(dj/stride)**2 ).astype(np.float32))
        edges.append([j,np.mod(j+dj,self.hidden_node)])
        edge_weights.append(np.exp( -(dj/stride)**2 ).astype(np.float32))
    encode_stride=3
    encoder_edges=[]
    decoder_edges=[]
    for j in range(self.hidden_node):
        center_grid = np.mod(int(j*self.ndim/self.hidden_node),self.ndim)
        for i in range(center_grid-encode_stride,center_grid+encode_stride):
            encoder_edges.append([np.mod(i,self.ndim),j])
            decoder_edges.append([j,np.mod(i,self.ndim)])

    self.edge_index=torch.tensor(edges).transpose(1,0)
    self.edge_weight=torch.tensor(edge_weights)
    self.encoder_edge_index=torch.tensor(encoder_edges).transpose(1,0)
    self.decoder_edge_index=torch.tensor(decoder_edges).transpose(1,0)

    self.encoder = GraphConv( (1,self.hidden_feature) , self.hidden_feature)
    self.acti    = relu
    self.predictor = GraphConv(self.hidden_feature, self.hidden_feature) ### PREDICTOR

    self.decoder = GraphConv( (self.hidden_feature,1) , 1)

  def forward(self, x):
    x=x.unsqueeze(-1)
    nbatch=x.shape[0]
    z=torch.rand(nbatch,self.hidden_node,self.hidden_feature)

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
    decoder_edge_index_b=batch_index(self.decoder_edge_index,nbatch,self.hidden_node,self.ndim)
    edge_index_b=batch_index(self.edge_index,nbatch,self.hidden_node,self.hidden_node)
    edge_weight_b=torch.cat([self.edge_weight for i in range(nbatch)], dim=0)

    x_b=torch.cat([x[i] for i in range(nbatch)], dim=0)
    z_b=torch.cat([z[i] for i in range(nbatch)], dim=0)
    z_b = self.acti(self.encoder((x_b,z_b),encoder_edge_index_b))

    z_b = self.acti(self.predictor(z_b,edge_index_b,edge_weight_b))
    y = torch.rand((nbatch,40,1))

    y_b=torch.cat([y[i] for i in range(nbatch)], dim=0)
    y = self.decoder((z_b,y_b),decoder_edge_index_b).view(nbatch,40)

    return y

model=nnmodel(40)
loss=nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(),lr=1.0e-3)

model_name="gcn"

x_smp=torch.rand((10,40))
print(x_smp.shape)
print(model(x_smp).shape)



