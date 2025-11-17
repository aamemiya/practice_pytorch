import os
import netCDF4 
import numpy as np
import numpy.linalg as LA             
import glob
import param

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

#import linear as custom
#import rnn as custom
#import cnn as custom
#import nn as custom
#import encoder_decoder as custom
#import transconv as custom
import gcn as custom
#import gcn_light as custom
#import convlstm as custom


# integration
ncdir = param.param_exp['expdir']

# load nature 
nc = netCDF4.Dataset('../' + ncdir + '/nature.nc','r',format='NETCDF4')
v = np.array(nc.variables['v'][:], dtype=type(np.float64)).astype(np.float32)
time = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

#nsmp=data_input.shape[0]
nsmp=4
nval=2

data_input=v[:nsmp-1,:]
data_output=v[1:nsmp,:]-v[:nsmp-1,:]

data_input_val=v[nsmp-1:nsmp+nval-1,:]
data_output_val=v[nsmp:nsmp+nval,:]-v[nsmp-1:nsmp+nval-1,:]

class MyDataset(TorchDataset):
    def __init__(self, *arrays, do_transform=True, do_transform_y=False):
        self.arrays = arrays
        self.do_transform = do_transform
        self.do_transform_y = do_transform_y
        self.x_mean = arrays[0].mean(axis=0)
        self.x_std = arrays[0].std(axis=0)
        self.y_mean = arrays[1].mean(axis=0)
        self.y_std = arrays[1].std(axis=0)

    def __len__(self):
        return len(self.arrays[0])

    def __getitem__(self, idx):
        items = [torch.as_tensor(a[idx]) for a in self.arrays]
        if self.do_transform:
            if items[0].ndim == self.x_mean.ndim + 1:
              items[0] = [(items[i] - self.x_mean) / self.x_std for i in range(items[0].shape[0])]
            else:
              items[0] = (items[0] - self.x_mean) / self.x_std 
        if self.do_transform_y:
            if items[1].ndim == self.y_mean.ndim + 1:
              items[1] = [(items[i] - self.y_mean) / self.y_std for i in range(items[1].shape[0])]
            else:
              items[1] = (items[1] - self.y_mean) / self.y_std 


        return items if len(items) > 1 else items[0]

def train_loop(dataloader, model, loss_fn, optimizer):
    size=len(dataloader.dataset)
    num_batchs=len(dataloader)
    model.train()
    loss_total=0
###    print("=== Train loss ===")
    for batch, (X,y) in enumerate(dataloader):
        pred=model(X)
        loss=loss_fn(pred,y)
        
        # Backpropagation 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

#        if batch % 10 == 0:
#            loss, current = loss.item(), batch* dataloader.batch_size + len(X)
#            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        loss_total += loss.item()
    loss_total /= num_batchs
    if dataloader.dataset.do_transform_y : 
        loss_total *= torch.mean(dataloader.dataset.y_std)
        loss_total = loss_total.item()
    return loss_total

def test_loop(dataloader, model, loss_fn):
    size=len(dataloader.dataset)
    num_batchs=len(dataloader)
    model.eval()
    loss_total=0
###    print("=== Test loss ===")
    with torch.no_grad():
        for batch, (X,y) in enumerate(dataloader):
            pred=model(X)
            loss=loss_fn(pred,y)
        
#            if batch % 10 == 0:
#               current = batch* dataloader.batch_size + len(X)
#               print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            loss_total += loss.item()
    loss_total /= num_batchs
###    print(f"loss_total: {loss_total:>7f}")
    if dataloader.dataset.do_transform_y : 
        loss_total *= torch.mean(dataloader.dataset.y_std)
        loss_total = loss_total.item()
    return loss_total


torch_data_input=torch.from_numpy(data_input.astype(np.float32))
torch_data_output=torch.from_numpy(data_output.astype(np.float32))

torch_dataset=MyDataset(torch_data_input,torch_data_output,do_transform_y=False)
torch_dataloader=DataLoader(torch_dataset,batch_size=32,shuffle=True)

#
#first_batch_x, first_batch_y =next(iter(torch_dataloader))

#for j in range(first_batch_y.shape[0]):
#  print(first_batch_y[j,0:4])
#quit()

torch_data_val_input=torch.from_numpy(data_input_val.astype(np.float32))
torch_data_val_output=torch.from_numpy(data_output_val.astype(np.float32))

torch_dataset_val=MyDataset(torch_data_val_input,torch_data_val_output,do_transform_y=False)
torch_dataloader_val=DataLoader(torch_dataset_val,batch_size=2,shuffle=True)

start_epoch=1


model=custom.model
loss=custom.loss
optimizer=custom.optimizer
model_name=custom.model_name

#model=nnmodel()
#loss=nn.MSELoss()
#optimizer= torch.optim.SGD(model.parameters(),lr=1.0e-3)


ckpt_list=glob.glob("./checkpoint/"+model_name+".ckpt.*")
if ckpt_list:
    ckpt_list_sort=sorted(ckpt_list, key=lambda x: int(x.split('.')[-1]))
    ckpt = torch.load(ckpt_list_sort[-1], weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch=ckpt['epoch']+1
    print("Load from latest checkpoint file:" +ckpt_list[-1])

total_epoch=100
log_intv=100
logs=[]
for epoch in range(start_epoch,total_epoch+start_epoch):
#    print(f"Epoch {epoch}\n-------------------------------")
    train_loss=train_loop(torch_dataloader,model,loss,optimizer)
    test_loss=test_loop(torch_dataloader_val,model,loss)

    if np.mod(epoch,log_intv) == 0 :
        ckptpath="./checkpoint/"+model_name+".ckpt."+str(epoch).zfill(3)
        torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_loss,
                    }
                ,ckptpath)
    print(f"Epoch: {epoch:3g},  train_loss: {train_loss:>7f}, test_loss: {test_loss:>7f}")
    logs.append([epoch, train_loss, test_loss])

logs=np.array(logs).transpose()
def plot_history( epoch_nums, loss, val_loss ,normalized=True):
		"""
		plot the history of the traning
		:param normalized: if True normalize both validation/traning loss to 1 fort he first eppoch.
		:return: the matplotlib figure
		"""
		fig,ax = plt.subplots()
		S1 = loss[2] if normalized else 1
		S2 = val_loss[2] if normalized else 1
		ax.semilogy(np.array(loss[2:])/S1, color='gray', linewidth=2, label='train')
		ax.semilogy(np.array(val_loss[2:])/S2, color='black', linewidth=2, label='test')
		ax.legend()
		return fig
fig=plot_history(logs[0],logs[1],logs[2],normalized=False)
fig.savefig('history.png')

print("Done.")
quit()

