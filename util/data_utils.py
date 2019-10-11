import os
from os.path import dirname, abspath

os.sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

from util import general_utils
import config
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
        

def get_subsequences(video, sequence_length=config.SEQUENCE_LENGTH):    
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
                
            
def read_sequence(sequence):
    frames = list()
    targets = list()
    
    frame_files, target_files = sequence
    
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)        
        frame = cv2.resize(frame, dsize=(config.INPUT_SIZE[1], config.INPUT_SIZE[0]))                        
        frame = np.rollaxis(frame, -1, 0)  # channel last -> channel first         
        frames.append(frame)
        
    for target_file in target_files:        
        target = cv2.imread(target_file, 0)                
        _, target = cv2.threshold(target, thresh=254, maxval=255, type=cv2.THRESH_BINARY)
        target = cv2.resize(target, dsize=(config.INPUT_SIZE[1], config.INPUT_SIZE[0]))
        target = (target > 0).astype("uint8")
        target = np.expand_dims(target, axis=-1)
        target = np.rollaxis(target, -1, 0)  # channel last -> channel first         
        targets.append(target)
        
    frames = np.expand_dims(frames, axis=1)
    targets = np.expand_dims(targets, axis=1)
    
    return frames, targets


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