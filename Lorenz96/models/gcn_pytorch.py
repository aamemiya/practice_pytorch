import os
import netCDF4 
import numpy as np
import numpy.linalg as LA             
import param

import matplotlib.pyplot as plt

import torch
from torch import nn

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv

from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData

# integration
ncdir = param.param_exp['expdir']

# load nature 
nc = netCDF4.Dataset('../' + ncdir + '/nature.nc','r',format='NETCDF4')
v = np.array(nc.variables['v'][:], dtype=type(np.float64)).astype(np.float32)
time = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

#nsmp=data_input.shape[0]
nsmp=3001

ndim=v.shape[-1]

data_input=v[:nsmp-1,:]
data_output=v[1:nsmp,:]-v[:nsmp-1,:]

### Graph representation of the input time series 

class MyListDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

stride=5
edges=[]
for j in range(ndim):
  for dj in range(1,stride-1):
    edges.append([j,np.mod(j-dj,ndim)])
    edges.append([j,np.mod(j+dj,ndim)])


edge_index=torch.tensor(edges).transpose(1,0)
pos=torch.tensor(np.linspace(0,ndim-1,ndim))

##data_input=Data(torch.from_numpy(data_input[0]).unsqueeze(axis=-1),edge_index=edge_index,pos=pos)
##data_output=Data(torch.from_numpy(data_output[0]).unsqueeze(axis=-1),pos=pos)

datalist_input=[]
for j in range(10): 
  data=HeteroData()
  data["src"].x=torch.from_numpy(data_input[j]).unsqueeze(axis=-1)
  data["src"].edge_index=edge_index
  data["dst"].x=torch.from_numpy(data_output[j]).unsqueeze(axis=-1)
  data["dst"].edge_index=edge_index
#  datalist_input.append(Data(x=torch.from_numpy(data_input[j]).unsqueeze(axis=-1),y=torch.from_numpy(data_output[j]).unsqueeze(axis=-1),edge_index=edge_index,pos=pos))
  datalist_input.append(data)

dataset_input=MyListDataset(datalist_input)
#print(len(dataset_input))
#print(dataset_input[1])
#quit()
#print(edge_index)
#print(pos)
#quit()
##print(data)

class GNNmodel_simple(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.processor = GraphConv(in_channels=(1,1),out_channels=1,add_self_loops=False) ### Version ?
        self.processor = GraphConv(in_channels=(1,1),out_channels=1)

    def forward(self, batch: HeteroData):
        x_s=batch["src"].x
        x_d=batch["dst"].x
        eidx=batch["src"].edge_index
        print(x_s.shape)
        print(eidx.shape)
        quit()
        Ns=batch["src"].num_nodes
        Nd=batch["dst"].num_nodes
        
        y = self.processor((x_s, x_d), eidx, size=(Ns, Nd))  # (sum Nd_i, 1)
#        x, edge_index = data.x, data.edge_index
#        y = self.processor(x, edge_index)
#        #return F.relu(y)
        return y

loader=DataLoader(dataset_input,batch_size=5,shuffle=False)
dataset_input_batch=next(iter(loader))
print(dataset_input_batch[1]["src"].x.shape)
#quit()
model=GNNmodel_simple()
#test_output=model(dataset_input[1:3])
test_output=model(dataset_input_batch)
print(test_output[1].shape)
quit()



class GNNmodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GraphConv(1,16)
        self.processor = GraphConv(16,16)
        self.decoder = GraphConv(16,1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.encoder((x,z), edge_index)
        z = F.relu(z)
        z = self.processor(z, edge_index)
        y = self.decoder(z, edge_index)
        return F.relu(y)


