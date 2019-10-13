import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import os
from os.path import dirname, abspath

os.sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

"""
ARGS
"""
import argparse

parser = argparse.ArgumentParser(description="Visualize outputs of a model")
parser.add_argument("--data_dir")
parser.add_argument("--dataset_dir")
parser.add_argument("--model_path")
parser.add_argument("--result_path")

args = parser.parse_args()

"""
"""
from shutil import copyfile

tmp_path = f"{args.dataset_dir}/{args.data_dir}"
copyfile(f"{tmp_path}/temporalROI.txt", f"{tmp_path}/temporalROI_test.txt")

"""
DATASET
"""
import torch
from torchvision import transforms

from dataset import VideoSequenceDataset
from transform import Rescale, ToTensor, Normalize
import config


video_dataset = VideoSequenceDataset(
    data_file="",
    data_dir=args.data_dir,    
    path_to_data=args.dataset_dir,
    transform=(transforms.Compose([
            Rescale((128, 128)),
            Normalize(),
            ToTensor()
        ])),
    sequence_length=None,
    # temporalROI="temporalROI_test",
    temporalROI="temporalROI_test"
)

test_loader = torch.utils.data.DataLoader(video_dataset,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=0)


"""
Test data:
cameraJitter/badminton
baseline/pedestrians
baseline/highway
badWeather/snowFall
"""


"""
MODEL
"""
from net import GMMNet
import torch.nn as nn

model = GMMNet(in_channels=3, 
               num_component=config.NUM_COMPONENT)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(config.DEVICE)


best_checkpoint = torch.load(args.model_path)
model.load_state_dict(best_checkpoint["state_dict"])
model.eval()



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
    
    with open(args.result_path, 'wb') as handle:
        pickle.dump(outputs[0, :, :, :, :].cpu().detach().numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)    
        
    break

print("Test loss:", loss.item())
            
print('=== TESTING: DONE. ===')