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
from torch_geometric.nn import GCNConv


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

stride=3
edges=[]
for j in range(ndim):
  for dj in range(1,stride-1):
    edges.append([j,np.mod(j-dj,ndim)])
    edges.append([j,np.mod(j+dj,ndim)])


edge_index=torch.tensor(edges)
pos=torch.tensor(np.linspace(0,ndim-1,ndim))

data=Data(torch.from_numpy(data_input[0]).unsqueeze(axis=-1),edge_index=edge_index,pos=pos)

print(edge_index)
print(pos)
print(data)

