import torch
from net.gmm_net import GMMNet
import config
import numpy as np

model = GMMNet(in_channels=3, 
               num_component=4)

model.to(config.DEVICE)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("#parameters: ", params)

frames = torch.Tensor(8, 15, 3, 128, 128).to(config.DEVICE)
targets = torch.Tensor(8, 15, 1, 128, 128).to(config.DEVICE)

outputs = model(frames, targets)
print(outputs.shape)