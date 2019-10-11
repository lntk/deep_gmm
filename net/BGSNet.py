import torch.nn as nn
import torch.nn.functional as F
import torch
from layer.convolution_lstm import ConvLSTM
from layer.encoder import Encoder
from torch.distributions.normal import Normal


class BGSNet(nn.Module):
    def __init__(self, in_channels, num_component, kernel_size):
        super(BGSNet, self).__init__()

        encoder_out_channels = 16                
        
        self.conv_lstm = ConvLSTM(in_channels=in_channels,
                                  num_component=num_component,
                                  kernel_size=kernel_size)
        
        self.num_component = num_component

    def forward(self, frames, targets):
        outputs = self.conv_lstm(frames)
        
        weights = torch.softmax(outputs[:, :, 0: self.num_component, :, :], dim=2)
        means = torch.sigmoid(outputs[: , :, self.num_component: 2*self.num_component, :, :])
                
        ## Gaussian
        # batch_size, sequence_length, _, height, width = outputs.shape
        # weights = torch.softmax(outputs[:, :, 0: self.num_component, :, :], dim=2)
        # means = outputs[: , :, self.num_component: 2*self.num_component, :, :]
        # stds = torch.exp(outputs[: , :, 2*self.num_component:, :, :])
        
        # max_weight_index = torch.argmax(weights, dim=2, keepdim=True)        
        # max_means = torch.index_select(means, dim=2, index=max_weight_index)
        # max_stds = torch.index_select(stds, dim=2, index=max_weight_index)
        
        # # Reparamterization trick
        # unit_gaussian = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        # epsilon = unit_gaussian.sample(sample_shape=(sequence_length, batch_size, 1, height, width))
        
        # # Sample from the largest component
        # probability_maps = epsilon * max_stds + max_means
        
        return weights, means, targets
