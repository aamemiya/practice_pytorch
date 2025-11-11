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
        for i in range(j*4-encode_stride,j*4+encode_stride):
            encoder_edges.append([np.mod(i,40),j])
            decoder_edges.append([j,np.mod(i,40)])

    self.edge_index=torch.tensor(edges).transpose(1,0)
    self.edge_weight=torch.tensor(edge_weights)
    self.encoder_edge_index=torch.tensor(encoder_edges).transpose(1,0)
    self.decoder_edge_index=torch.tensor(decoder_edges).transpose(1,0)


    #self.encoder = nn.Sequential(
    #        nn.Linear(40, self.hidden_dim), ### ENCODER
    #        nn.ReLU(),
    #        )
    self.encoder = GraphConv( (1,self.hidden_feature) , self.hidden_feature)

    self.predictor = GraphConv(self.hidden_feature, self.hidden_feature) ### PREDICTOR

    #self.decoder = nn.Sequential(
    #        nn.Linear(self.hidden_dim,40),    ### DECODER
    #        nn.ReLU(),
    #)
    self.decoder = GraphConv( (self.hidden_feature,1) , 1)

  def forward(self, x):
    x=x.unsqueeze(-1)
    nbatch=x.shape[0]
    z=torch.from_numpy(np.random.rand(nbatch,self.hidden_node,self.hidden_feature).astype(np.float32))

#    print(x.shape)
#    print(z.shape)
#    quit()

    xz_data=[Data(x=(x[i],z[i]), edge_index=self.encoder_edge_index) for i in range(nbatch)]
    xz_batch=Batch.from_data_list(xz_data)
    z = self.encoder(xz_batch.x,xz_batch.edge_index)
#    y = self.encoder(x.unsqueeze(-1))
#    y=y.reshape(nbatch,self.hidden_node,self.hidden_feature)

    z_data=[Data(x=z[i], edge_index=self.edge_index, edge_weight=self.edge_weight) for i in range(nbatch)]
    z_batch=Batch.from_data_list(z_data)
#    print(nbatch)
#    print(y_batch.view)
#    print(y_batch.batch)
#    quit()
    z = self.predictor(z_batch.x,z_batch.edge_index,edge_weight=y_batch.edge_weight).view(nbatch,self.hidden_dim)
#    print(y.view)
#    quit()
    y = np.random.rand(nbatch,40,1)
    zy_data=[Data(x=(z[i],y[i]), edge_index=self.decoder_edge_index) for i in range(nbatch)]
    zy_batch=Batch.from_data_list(zy_data)
    y = self.decoder(zy_batch.x,zy_batch.edge_index).squeeze(-1)
    return y

model=nnmodel()
loss=nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(),lr=1.0e-2)

model_name="gcn"

x_smp=torch.from_numpy(np.random.rand(10,40).astype(np.float32))
print(x_smp.shape)
print(model(x_smp).shape)



