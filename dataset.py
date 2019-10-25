from torch.utils.data import Dataset
from skimage import io
import numpy as np
import torch

from util import general_utils, data_utils
from data.CDnet2014_data_utils import get_data_files


class VideoSequenceDataset(Dataset):
    def __init__(self, path_to_data, sequence_length, data_file=None, transform=None, temporalROI="temporalROI", data_dirs=None):
        if data_dirs is None:
            if data_file is None:
                raise Exception("Unspecified data.")            
            data_dirs = general_utils.read_lines(data_file)
        else:
            data_dirs = data_dirs
        data = get_data_files(data_dirs, path_to_data=path_to_data, temporalROI=temporalROI)  
        
        sequences = data_utils.get_sequences(data=data, sequence_length=sequence_length)
        
        self.frame_sequences = sequences["inputs"]
        self.target_sequences = sequences["groundtruths"]
        self.transform = transform
        
    
    def __len__(self):
        return len(self.frame_sequences)
    
    
    def __getitem__(self, idx):
        frame_files = self.frame_sequences[idx]
        target_files = self.target_sequences[idx]
        
        frames = [io.imread(frame_file) for frame_file in frame_files]
        targets = [np.expand_dims(np.where(np.isclose(io.imread(target_file, as_gray=True), 255.0), 1.0, 0.0), axis=-1) for target_file in target_files]
        
        if self.transform:
            frames, targets = self.transform((frames, targets))
        
        return frames, targets
    
    
    def view_batch(self):
        """
        View the first 5 frames of a random sequence
        """
        
        i = np.random.randint(self.__len__)
        
        raise NotImplementedError
        