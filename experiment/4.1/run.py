import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

import os
from os.path import dirname, abspath

os.sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

"""
DATASET
"""
from dataset import VideoSequenceDataset
from transform import Rescale, ToTensor, Normalize
from torchvision import transforms
import torch
import config

video_dataset = VideoSequenceDataset(
    # data_file="experiment/4/toy_train.txt",
    # path_to_data="data/toy",
    data_file='experiment/4.1/CDnet2014_train.txt',
    path_to_data='/content/dataset',
    transform=(transforms.Compose([
            Rescale((128, 128)),
            Normalize(),
            ToTensor()
        ])),
    sequence_length=config.SEQUENCE_LENGTH
)

train_loader = torch.utils.data.DataLoader(video_dataset,
                                           batch_size=(config.BATCH_SIZE),
                                           shuffle=True,
                                           num_workers=0)

"""
Train data:
baseline/highway
baseline/office
baseline/pedestrians
baseline/PETS2006
"""


"""
MODEL
"""
from net import GMMNet
import torch.nn as nn
import numpy as np

model = GMMNet(in_channels=3, 
               num_component=config.NUM_COMPONENT)

print(model)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("#parameters: ", params)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(config.DEVICE)


"""
OPTIMIZER
"""
import torch.optim as optim

optimizer = optim.Adam((model.parameters()), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


"""
TRAINING
"""
from tqdm import tqdm
import time, visualize
from loss import jaccard_loss, binary_cross_entropy
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision

writer = SummaryWriter(log_dir="experiment/4.1/tensorboard")

def save_checkpoint(state, filename):
    torch.save(state, filename)    

print('=== TRAINING: STARTED. ===')


best_loss = 1e10

for epoch in range(config.NUM_EPOCH):
    running_loss = 0.0
    running_iou = 0.0    
    
    batch_id = 0
    for frames, targets in tqdm(train_loader, ncols=100):        
        batch_id += 1
        frames = frames.to(config.DEVICE)
        targets = targets.to(config.DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(frames, targets)
                
        # loss = binary_cross_entropy(outputs, targets)
        loss = jaccard_loss(outputs, targets)
        
        loss.backward()
        optimizer.step()
                                
        running_loss += loss.item()
        
    if running_loss < best_loss:
        best_loss = running_loss
        save_checkpoint({
            "epoch": epoch,        
            "state_dict": model.state_dict(),
            "best_loss": best_loss,
            "optimizer" : optimizer.state_dict(),
        }, filename=f"experiment/4.1/best_model_{best_loss}.pth")   
        
    
    writer.add_scalar('loss', running_loss, epoch)         
    
    print(f"Epoch {epoch} - Loss {running_loss}")
    
writer.close()    

print('=== TRAINING: DONE. ===')
