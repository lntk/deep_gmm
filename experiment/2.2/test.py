import os
from os.path import dirname, abspath

os.sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

import torch
from torchvision import transforms

from dataset import VideoSequenceDataset
from transform import Rescale, ToTensor, Normalize
import config

"""
DATASET
"""
video_dataset = VideoSequenceDataset(
    data_file='data/toy_test.txt',
    path_to_data='data/toy',
    transform=(transforms.Compose([
            Rescale(128),
            Normalize(),
            ToTensor()
        ])),
    sequence_length=None
)

test_loader = torch.utils.data.DataLoader(video_dataset,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=0)


"""
MODEL
"""
model = torch.load("experiment/2.2/model.pth")
model.eval()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

model.to(config.DEVICE)


"""
TESTING PHASE
"""
from tqdm import tqdm
from skimage import io
from loss import jaccard_loss
import pickle

print('=== TESTING: STARTED. ===')
for frames, targets in tqdm(test_loader, ncols=100):        
    frames = frames.to(config.DEVICE)
    targets = targets.to(config.DEVICE)
    
    outputs = model(frames, targets)   
    B, S, _, H, W = outputs.shape
    
    loss = jaccard_loss(outputs, targets)
    
    # for i in range(S):
    #     # io.imsave(f"experiment/2.2/result/{'{:06d}'.format(i + 1)}.jpg", (outputs[0, i, 0, :, :] > config.TEST_THRESHOLD).long().cpu().numpy())            
    #     io.imsave(f"experiment/2.2/result/{'{:06d}'.format(i + 1)}.jpg", outputs[0, i, 0, :, :].cpu().detach().numpy())            
    
    with open('experiment/2.2/result.pkl', 'wb') as handle:
        pickle.dump(outputs[0, :, :, :, :].cpu().detach().numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)    
        
    break

print("Test loss:", loss.item() / S)
            
print('=== TESTING: DONE. ===')