import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import numpy as np 
from torch import nn

quicklook=False
quicklook_test=True

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

if quicklook == True:
  figure = plt.figure(figsize=(8, 12))
  cols, rows = 3, 4
  for i in range(1, cols * rows + 1):
      sample_idx = torch.randint(len(training_data), size=(1,)).item()
      img, label = training_data[sample_idx]
      figure.add_subplot(rows, cols, i)
      plt.title(str(label))
      plt.axis("off")
      plt.imshow(img.squeeze(), cmap="gray")
  plt.savefig("training_data.png")

train_dataloader=DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader=DataLoader(test_data, batch_size=64, shuffle=True)

#img, label = training_data[0]

#print(type(label))
#print(label.shape)
#quit()
#label_np=label.numpy()
#label_np=label
#img_np=img.squeeze().numpy()
#print(img_np)
#print(label_np)
#quit()

#train_img, train_label = next(iter(train_dataloader))
#train_img_np=train_img[0].squeeze().numpy()
#print(train_img_np.shape)
#print(train_img_np)

class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
    )
  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model=NeuralNetwork().to(device)
#model=NeuralNetwork()
model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
#print(model)
#img, label= test_data[0]
#print(model(img).squeeze(),label)
#print(model(img).squeeze().argmax(0).item())
#quit()

### Hyper_params 
learning_rate = 1e-3 
batch_size = 64
epochs = 5 


loss_fn = nn.CrossEntropyLoss()


optimizer= torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()
            correct += (pred.argmax(1) == y.to(device).argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs): 
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

model=model.to("cpu")

if quicklook_test == True:
  figure = plt.figure(figsize=(8, 12))
  cols, rows = 3, 4
  for i in range(1, cols * rows + 1):
      sample_idx = torch.randint(len(test_data), size=(1,)).item()
      img, label = test_data[sample_idx]
      figure.add_subplot(rows, cols, i)
      with torch.no_grad():
        label_pred=model(img).squeeze().argmax(0).item()
        label_item=label.squeeze().argmax(0).item()
        print(label_item,label_pred)
      plt.title(str(label_item)+","+str(label_pred))
      plt.axis("off")
      plt.imshow(img.squeeze(), cmap="gray")
  plt.savefig("test_final.png")


torch.save(model.state_dict(), 'model_weights.pth')

#model_new=NeuralNetwork()

#model_new.load_state_dict(torch.load("model_weights.pth", weights_only=True))
#model_new.eval()
#print(model_new.named_parameters()[0])
