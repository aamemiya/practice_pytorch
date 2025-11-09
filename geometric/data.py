import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset

fcount=0

def draw_graph(data_input):
  global fcount
  data_nx=to_networkx(data_input, to_undirected=True)
  pos=nx.spring_layout(data_nx)
  fig, ax = plt.subplots(figsize=(6,6))
  nx.draw(data_nx, pos, with_labels=True, node_color="lightblue",edge_color="gray",node_size=600,font_size=8) 
  ax.set_title(str(data_input["y"].item()),fontsize=12)
  fig.savefig("graph_"+str(fcount).zfill(3)+".png")
  fcount+=1

#edge_index = torch.tensor([[0, 1, 1, 2],
#                           [1, 0, 2, 1]], dtype=torch.long)
#x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
#position=torch.tensor( [[-1,0], [0,1], [1,0]] )
#data = Data(x=x, edge_index=edge_index,pos=position)
#print(data)

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
perm=torch.randperm(len(dataset))
for j in range(5):
  draw_graph(dataset[perm[j]])

loader=DataLoader(dataset, batch_size=32, shuffle=True)


