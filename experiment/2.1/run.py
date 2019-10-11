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
    data_file='data/toy_train.txt',
    path_to_data='data/toy',
    transform=(transforms.Compose([
            Rescale(128),
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

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model)

model.to(config.DEVICE)


"""
OPTIMIZER
"""
import torch.optim as optim

# optimizer = optim.Adam((model.parameters()), lr=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)



"""
CLEAR CACHE
"""
from util import general_utils

general_utils.delete_files_in_dir("tmp")


"""
TRAINING
"""
from tqdm import tqdm
import time, visualize
from loss import jaccard_loss, binary_cross_entropy, JaccardLoss


print('=== TRAINING: STARTED. ===')
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
                
        loss = JaccardLoss()(outputs, targets)
        
        loss.backward()
        optimizer.step()
                
        running_loss += loss.item()
        
    print(f"Epoch {epoch} - Loss {running_loss}")

print('=== TRAINING: DONE. ===')
