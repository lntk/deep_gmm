import torch
from net import BGSNet
from loss import negative_log_likelihood_loss, mean_iou_score
import config
from sampling import *
from util import image_utils
import numpy as np
from dataset import VideoSequenceDataset
from torchvision import transforms
from transform import Rescale, ToTensor, Normalize
import torch, config
from tqdm import tqdm

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
                                           num_workers=4)

model = BGSNet(in_channels=3, 
               num_component=(config.NUM_COMPONENT),
               kernel_size=3)

model.to(config.DEVICE)

import torch.optim as optim
optimizer = optim.Adam((model.parameters()), lr=0.001)

import tqdm, time, visualize

print('=== TRAINING: STARTED. ===')
for epoch in range(config.NUM_EPOCH):
    running_loss = 0.0
    running_iou = 0.0
    batch_counter = 0
    
    for frames, targets in tqdm(train_loader):
        batch_counter += 1        
        print(f"= batch {batch_counter}")
        
        frames = frames.to(config.DEVICE)
        targets = targets.to(config.DEVICE)
        
        optimizer.zero_grad()
        
        start = time.time()
        weights, means, targets = model(frames, targets)
        end = time.time()
        print(f"=== batch forward time: {end - start} s")
        
        # activations = list()
        # names = list()
        
        # activations += [weights[0, 0, i, :, :] for i in range(config.NUM_COMPONENT)]
        # names += [f"weights_{i}" for i in range(config.NUM_COMPONENT)]
        # activations += [means[0, 0, i, :, :] for i in range(config.NUM_COMPONENT)]
        # names += [f"means_{i}" for i in range(config.NUM_COMPONENT)]
        
        # visualize.see_activations(activations=activations, names=names, normalize=True, save=f"tmp/epoch_{epoch}_batch_{batch_counter}.png")
        
        
        start = time.time()
        target_samples = sequence_sample_bernoulli_mixtures(weights, means, channel_first=True)
        end = time.time()
        print(f"=== batch sample time: {end - start} s")
        
        loss = negative_log_likelihood_loss(weights, means, targets)
        
        start = time.time()
        loss.backward()
        optimizer.step()
        end = time.time()
        print(f"=== batch backward time: {end - start} s")
        
        start = time.time()
        running_loss += loss.item()
        iou_score = 0.0
        batch_size = targets.shape[0]
        target_lists = [[targets[i, j, 0, :, :] for j in range(config.SEQUENCE_LENGTH)] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(config.SEQUENCE_LENGTH):
                iou_score += mean_iou_score(target_samples[i][j], target_lists[i][j].cpu().long())

        iou_score = iou_score / (config.SEQUENCE_LENGTH * batch_size)
        running_iou += iou_score
        end = time.time()
        print(f"=== batch statistics time: {end - start} s")

    print(f"Epoch {epoch} - Loss {running_loss} - Mean IOU {running_iou / len(train_loader)}")

print('=== TRAINING: DONE. ===')
