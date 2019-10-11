from skimage import transform
import numpy as np
import torch


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, sample):
        frames, targets = sample

        h, w = frames[0].shape[:2]
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        frames = [transform.resize(frame, (new_h, new_w), preserve_range=True) for frame in frames]
        targets = [transform.resize(target, (new_h, new_w)) for target in targets]        
        
        frames = np.array(frames)
        targets = np.array(targets)

        return frames, targets
    
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        frames, targets = sample
        
        # swap color axis because
        # numpy sequences: S x H x W x C
        # torch sequences: S x C X H X W
        frames = frames.transpose((0, 3, 1, 2))
        targets = targets.transpose((0, 3, 1, 2))
        
        return torch.from_numpy(frames).float(), torch.from_numpy(targets).float()
    

class Normalize(object):
    """Normalize frames into range (0, 1)."""

    def __call__(self, sample):    
        frames, targets = sample
        
        frames = frames / 255.
        
        return frames, targets