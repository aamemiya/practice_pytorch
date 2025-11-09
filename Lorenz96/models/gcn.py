import os
import numpy as np

import torch
from torch import nn
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.data import Data, Batch

class nnmodel(nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden_node=10
    self.hidden_feature=8
    self.hidden_dim=self.hidden_node*self.hidden_feature

    stride=3
    edges=[]
    edge_weights=[]
    for j in range(self.hidden_node):
      for dj in range(1,stride-1):
        edges.append([j,np.mod(j-dj,self.hidden_node)])
        edge_weights.append(np.exp( -(dj/stride)**2 ).astype(np.float32))
        edges.append([j,np.mod(j+dj,self.hidden_node)])
        edge_weights.append(np.exp( -(dj/stride)**2 ).astype(np.float32))
    self.edge_index=torch.tensor(edges).transpose(1,0)
    self.edge_weight=torch.tensor(edge_weights)

    self.encoder = nn.Sequential(
            nn.Linear(40, self.hidden_dim), ### ENCODER
            nn.ReLU(),
            )
    self.predictor = GraphConv(self.hidden_feature, self.hidden_feature) ### PREDICTOR
    self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim,40),    ### DECODER
            nn.ReLU(),
    )
  def forward(self, x):
    y = self.encoder(x)
    nbatch=y.shape[0]
    y=y.reshape(nbatch,self.hidden_node,self.hidden_feature)
    y_data=[Data(x=y[i], edge_index=self.edge_index, edge_weight=self.edge_weight) for i in range(nbatch)]
    y_batch=Batch.from_data_list(y_data)
#    print(nbatch)
#    print(y_batch.view)
#    print(y_batch.batch)
#    quit()
    y = self.predictor(y_batch.x,y_batch.edge_index,edge_weight=y_batch.edge_weight).view(nbatch,self.hidden_dim)
#    print(y.view)
#    quit()
    y = self.decoder(y)
    return y

model=nnmodel()
loss=nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(),lr=1.0e-2)

model_name="gcn"


