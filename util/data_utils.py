import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

from util import general_utils
from data.CDnet2014_data_utils import get_data_files


def get_sequences(data, sequence_length):        
    videos = list(data.keys())
    all_subsequences = {"inputs": list(), "groundtruths": list()}
        
    for video in videos:
        if sequence_length is None:
            subsequences = get_all_frames(data[video])
        else:
            subsequences = get_subsequences(data[video], sequence_length=sequence_length)
            
        all_subsequences["inputs"] += subsequences["inputs"]
        all_subsequences["groundtruths"] += subsequences["groundtruths"]
        
    return all_subsequences
        

def get_subsequences(video, sequence_length=20):    
    num_frame = len(video["inputs"])    
    num_subsequence = int(num_frame / sequence_length)
    
    subsequences = {"inputs": list(), "groundtruths": list()}
    for i in range(num_subsequence):
        subsequences["inputs"].append(video["inputs"][i * sequence_length: (i + 1) * sequence_length])
        subsequences["groundtruths"].append(video["groundtruths"][i * sequence_length: (i + 1) * sequence_length])
    
    
    # Padding    
    if num_frame == num_subsequence * sequence_length:
        shortage = 0 
    else:
        shortage = sequence_length - (num_frame - num_subsequence * sequence_length)                    
        
        indices = list(range((i + 1) * sequence_length, num_frame))        
        padded_indices = np.pad(indices, pad_width=(0, shortage), mode="edge")

        subsequences["inputs"].append([video["inputs"][i] for i in padded_indices])        
        subsequences["groundtruths"].append([video["groundtruths"][i] for i in padded_indices])
        
        
    return subsequences


def get_all_frames(video):
    sequences = {"inputs": list(), "groundtruths": list()}
    sequences["inputs"].append(video["inputs"])
    sequences["groundtruths"].append(video["groundtruths"])
    
    return sequences                            


def view_random_sample(frames, targets):
    sequence_length, batch_size, _, _, _ = frames.shape
    seg_id = random.choice(list(range(sequence_length)))
    batch_id = random.choice(list(range(batch_size)))
    
    frame = frames[seg_id, batch_id, :, :, :]
    frame = np.moveaxis(frame, source=0, destination=-1)    
    
    target = targets[seg_id, batch_id, 0, :, :]  
    
    print(np.unique(target))      
    
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(frame)        
    axs[1].imshow(target, cmap="gray")
    plt.show()