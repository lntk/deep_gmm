import torch
from torch.distributions.categorical import Categorical
import numpy as np

def sequence_sample_bernoulli_mixtures(weights, means, channel_first=False):                
    batch_size, sequence_length, _, _, _ = weights.shape
    
    sampled = list()
    
    for i in range(batch_size):    
        batch_samples = list()
        for j in range(sequence_length):
            curr_weights = weights[i, 0, :, :, :]
            curr_means = means[i, 0, :, :, :]    
            
            if channel_first:
                curr_weights = moveaxis_torch(curr_weights.detach().cpu(), source=0, destination=-1)
                curr_means = moveaxis_torch(curr_means.detach().cpu(), source=0, destination=-1)        
            
            sample = sample_bernoulli_mixtures(curr_weights, curr_means)
            batch_samples.append(sample)
        
        sampled.append(batch_samples)
    
    return sampled

def sample_bernoulli_mixtures(weights, means):
    sampled_components = Categorical(weights).sample()
    sampled_means = select_values_by_indices(means, sampled_components)
    
    sample = (sampled_means > 0.5).long()
    
    # bernoulli = Categorical(torch.stack([sampled_means, 1 - sampled_means], dim=-1))
    # sample = bernoulli.sample()
    
    return sample
    
def select_values_by_indices(a, indices):
    channel = a.shape[-1]
    duplicate_indices = torch.stack([indices for _ in range(channel)], dim=-1)
    
    duplicate_chosen = torch.gather(a, dim=-1, index=duplicate_indices)
    return duplicate_chosen[:, :, 0]
    
    
def moveaxis_torch(x, source, destination):
    x = x.numpy()
    x = np.moveaxis(x, source=source, destination=destination)
    return torch.from_numpy(x)