import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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

num_train=int(0.8*len(dataset))
train_loader=DataLoader(dataset[:num_train], batch_size=32, shuffle=True)
test_loader=DataLoader(dataset[num_train:], batch_size=32, shuffle=True)

#print(dataset.num_nodes)
#print(dataset.num_features)
#print(dataset.num_node_features)
#quit()

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)

print(dataset[0])
#print(dataset[0].shape)
print(model(dataset[0]).shape)
quit()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train_loop (dataloader, model, optimizer): 
    size=len(dataloader.dataset)
    model.train()
    for batch, data in enumerate(dataloader): 
       pred=model(data)
       loss=F.nll_loss()
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()

       if batch % 100 == 0:
           loss, current = loss.item(), batch * batch_size + len(X)
           print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(dataset[0].train_mask)
quit()

model.train()
for epoch in range(20):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
#>>> Accuracy: 0.8150

