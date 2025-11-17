import sys
import torch 


ckptfile=sys.argv[1]

ckpt = torch.load(ckptfile, weights_only=True)
print("loss = ", ckpt["loss"])
