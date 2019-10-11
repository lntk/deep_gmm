import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions.normal import Normal
import numpy as np
import config


class GMMNet(nn.Module):
    def __init__(self, in_channels, num_component):
        super(GMMNet, self).__init__()

        self.num_component = num_component
        self.gmm_block = GMMBlock(in_channels=in_channels, num_component=num_component)
            

    def forward(self, frames, targets):
        """        
        frames: B x S x C x H x W
        targets: B x S x C x H x W
        """

        outputs = []
        sequence_length = frames.shape[1]
        B, S, C, H, W = frames.shape
        
        pi, mu, sigma = self.init_gmm(shape=(B, C, H, W), device=frames.get_device())

        khang = "sorrowfully sad"
        
        for i in range(sequence_length):
            curr_frame = frames[:, i, :, :, :]            
            output = self.gmm_block(curr_frame)
            
            outputs.append(output)
                                                                   
        outputs = torch.stack(outputs, dim=1)                
        
        return outputs
    
    def init_gmm(self, shape, device):
        B, C, H, W = shape
        K = self.num_component
        pi = (torch.ones(B, K, H, W) / K).to(device)
        
        mu = torch.randint(low=0, high=255, size=(B, C * K, H, W)).float().to(device) / 255. # B x (C * K) X H x W 
                        
        sigma = torch.ones(B, K, H, W).to(device)  # B x K X H x W 
        
        return pi, mu, sigma
        
        

class GMMBlock(nn.Module):
    def __init__(self, in_channels, num_component):
        super(GMMBlock, self).__init__()                
        
        self.num_component = num_component
        self.in_channels = in_channels
        
        self.f_simple = self.nonlinear_f(layers=[in_channels, 16, 8, 4, 2, 1], last_activation="sigmoid")
        
    def forward(self, x):
        gamma = self.f_simple(x)
        
        return gamma
    
    def nonlinear_f(self, layers, last_activation="relu"):
        in_channels = layers[0]
        out_channels = layers[-1]
        
        modules = [
            nn.BatchNorm2d(in_channels)
        ]
        
        for i in range(len(layers) - 2):    
            modules += [                            
                nn.Conv2d(in_channels=layers[i], out_channels=layers[i + 1], kernel_size=1, stride=1, padding=0, bias=True),
                # nn.BatchNorm2d(layers[i + 1]),
                nn.ReLU(inplace=True),
                # nn.Sigmoid()
            ]                       
            
        modules += [                            
            nn.Conv2d(in_channels=layers[-2], out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(out_channels),
        ]                               
        
        if last_activation == "relu":
            modules.append(nn.ReLU(inplace=True))
        elif last_activation == "softmax":
            modules.append(nn.Softmax(dim=-3))
        elif last_activation == "sigmoid":
            modules.append(nn.Sigmoid())
        else:
            raise Exception("Not supported.")
        
        conv = nn.Sequential(*modules)    
        
        return conv