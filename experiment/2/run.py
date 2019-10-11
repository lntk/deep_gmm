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
        ]))
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

model = GMMNet(in_channels=3, 
               num_component=config.NUM_COMPONENT)

print(model)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

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
from loss import jaccard_loss


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
        
        preds = model(frames, targets)
        
        # import visualize
        # activations = [preds[0, i, 0, :, :] for i in range(config.SEQUENCE_LENGTH)]
        # names = [f"{i}" for i in range(config.SEQUENCE_LENGTH)]
        # visualize.see_activations(activations=activations, names=names, normalize=False, save=f"tmp/epoch_{epoch}_batch_{batch_id}.png")
                
        loss = jaccard_loss(preds, targets)
        
        loss.backward()
        optimizer.step()
                
        running_loss += loss.item()
        
    print(f"Epoch {epoch} - Loss {running_loss}")

print('=== TRAINING: DONE. ===')
