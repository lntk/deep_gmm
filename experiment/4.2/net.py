import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions.normal import Normal
import numpy as np
import config


class GMMNet(nn.Module):
    def __init__(self, in_channels, num_component, blocks=(8, 16, 32, 64), init_params=None):        
        super(GMMNet, self).__init__()

        self.blocks = blocks
        num_block = len(self.blocks)
        
        self.num_component = num_component
        
        self.gmm_block = GMMBlock(in_channels=self.blocks[-1], num_component=num_component)
        
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=blocks[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(blocks[0]),
            nn.ReLU6(inplace=True)         
        )        
                
        self.conv_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.blocks[0], out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),            
            nn.Sigmoid()
        )     
                
        self.downsamples = nn.ModuleList(
            [
                Downsample(in_channels=self.blocks[i], out_channels=self.blocks[i + 1]) 
                for i in range(num_block - 1)
            ]
        )
        self.upsamples = nn.ModuleList(
            [
                Upsample(in_channels=self.blocks[i], out_channels=self.blocks[i - 1]) 
                for i in range(1, num_block)
            ]
        )
        
        self.init_params = init_params
        
            
    def forward(self, frames, targets):
        """        
        frames: B x S x C x H x W
        targets: B x S x C x H x W
        """

        outputs = []
        sequence_length = frames.shape[1]
        B, S, C, H, W = frames.shape
                
        if self.init_params is None:
            if self.blocks == (8,):
                pi, mu, sigma = self.init_gmm(shape=(B, 8, 256, 256), device=frames.get_device())
            elif self.blocks == (8, 16):
                pi, mu, sigma = self.init_gmm(shape=(B, 16, 128, 128), device=frames.get_device())
            elif self.blocks == (8, 16, 32):
                pi, mu, sigma = self.init_gmm(shape=(B, 32, 64, 64), device=frames.get_device())
            elif self.blocks == (8, 16, 32, 64):
                pi, mu, sigma = self.init_gmm(shape=(B, 64, 32, 32), device=frames.get_device())
            else:
                raise NotImplementedError           
        else:
            pi, mu, sigma = self.init_params
            
        # if self.blocks == (8,):
        #     pi, mu, sigma = self.init_gmm(shape=(B, 8, 128, 128), device=frames.get_device())
        # elif self.blocks == (8, 16):
        #     pi, mu, sigma = self.init_gmm(shape=(B, 16, 64, 64), device=frames.get_device())
        # elif self.blocks == (8, 16, 32):
        #     pi, mu, sigma = self.init_gmm(shape=(B, 32, 32, 32), device=frames.get_device())
        # else:
        #     raise NotImplementedError
        
        for i in range(sequence_length):
            curr_frame = frames[:, i, :, :, :]
            assert curr_frame.shape == torch.Size(np.array([B, C, H, W]))            
            
            encode_0 = self.conv_encoder(curr_frame)
            assert encode_0.shape == torch.Size(np.array([B, self.blocks[0], H, W]))
                        
            encode_1 = self.downsamples[0](encode_0)
            assert encode_1.shape == torch.Size(np.array([B, self.blocks[1], int(H/2), int(W/2)]))            
            
            encode_2 = self.downsamples[1](encode_1)
            assert encode_2.shape == torch.Size(np.array([B, self.blocks[2], int(H/4), int(W/4)]))            
            
            encode_3 = self.downsamples[2](encode_2)
            assert encode_3.shape == torch.Size(np.array([B, self.blocks[3], int(H/8), int(W/8)]))            
            
            # pi, mu, sigma, encode_0_map = self.gmm_block(encode_0, pi, mu, sigma)
            # pi, mu, sigma, encode_1_map = self.gmm_block(encode_1, pi, mu, sigma)    
            # pi, mu, sigma, encode_2_map = self.gmm_block(encode_2, pi, mu, sigma)            
            pi, mu, sigma, encode_3_map = self.gmm_block(encode_3, pi, mu, sigma)            

            encode_2_map = self.upsamples[2](encode_3_map * encode_3)
            assert encode_2_map.shape == torch.Size(np.array([B, 1, int(H/4), int(W/4)]))            
                                                                     
            encode_1_map = self.upsamples[1](encode_2_map * encode_2)
            assert encode_1_map.shape == torch.Size(np.array([B, 1, int(H/2), int(W/2)]))            
            
            encode_0_map = self.upsamples[0](encode_1_map * encode_1)
            assert encode_0_map.shape == torch.Size(np.array([B, 1, H, W]))                                    
            
            output = self.conv_decoder(encode_0_map * encode_0)
            assert output.shape == torch.Size(np.array([B, 1, H, W]))
                                                
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
                        
        self.f_pi    = self.nonlinear_f(layers=[2 * num_component, 8, num_component], last_activation="softmax")        
        self.f_mu    = self.nonlinear_f(layers=[in_channels + in_channels * num_component + num_component, 16, in_channels * num_component])
        self.f_sigma = self.nonlinear_f(layers=[in_channels + in_channels * num_component + num_component, 8, num_component])        
        
        self.f_gamma = self.nonlinear_f(layers=[num_component, 4, 1], last_activation="sigmoid")        
        
    
    def forward(self, x, pi, mu, sigma):
        B, C, H, W = x.shape
        K = self.num_component        
        channel_dim = 1                
        
        assert pi.shape == torch.Size(np.array([B, K, H, W]))
        assert mu.shape == torch.Size(np.array([B, C * K, H, W]))
        assert sigma.shape == torch.Size(np.array([B, K, H, W]))        
        
        alpha = pi * torch.cat([self.multivariate_Gaussian(x, mu[:, i * C: (i + 1) * C, :, :], sigma[:, i: i+1, :, :]) for i in range(self.num_component)], dim=channel_dim)
        assert alpha.shape == torch.Size(np.array([B, K, H, W]))    
        
        rho = alpha * torch.cat([self.multivariate_Gaussian(x, mu[:, i * C: (i + 1) * C, :, :], sigma[:, i: i+1, :, :]) for i in range(K)], dim=channel_dim)
        assert rho.shape == torch.Size(np.array([B, K, H, W]))        
        
        pi = self.f_pi(torch.cat([pi, alpha], dim=channel_dim))
        assert pi.shape == torch.Size(np.array([B, K, H, W]))        
         
        mu = self.f_mu(torch.cat([x, mu, rho], dim=channel_dim))
        assert mu.shape == torch.Size(np.array([B, C * K, H, W]))        
        
        sigma = torch.exp(self.f_sigma(torch.cat([x, mu, rho], dim=channel_dim)))
        assert sigma.shape == torch.Size(np.array([B, K, H, W]))     

        gamma = self.f_gamma(pi * torch.cat([self.multivariate_Gaussian(x, mu[:, i * C: (i + 1) * C, :, :], sigma[:, i: i+1, :, :]) for i in range(self.num_component)], dim=channel_dim))        
        assert gamma.shape == torch.Size(np.array([B, 1, H, W]))
        
        return pi, mu, sigma, gamma
        
    
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
                # nn.ReLU6(inplace=True),
                nn.Sigmoid()
            ]                       
            
        modules += [                            
            nn.Conv2d(in_channels=layers[-2], out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(out_channels),
        ]                               
        
        if last_activation == "relu":
            modules.append(nn.ReLU6(inplace=True))
        elif last_activation == "softmax":
            modules.append(nn.Softmax(dim=-3))
        elif last_activation == "sigmoid":
            modules.append(nn.Sigmoid())
        else:
            raise Exception("Not supported.")
        
        conv = nn.Sequential(*modules)    
        
        return conv
    
    @staticmethod
    def multivariate_Gaussian(x, mu, sigma):
        B, C, H, W = x.shape
        assert mu.shape == torch.Size(np.array([B, C, H, W]))
        assert sigma.shape == torch.Size(np.array([B, 1, H, W]))
                
        sigma_square = torch.pow(sigma, 2)        
        
        distance = torch.sum(torch.pow(x - mu, 2), dim=1, keepdim=True)
        
        density = 1/(torch.pow(2 * np.pi * sigma_square, C / 2)) * torch.exp(- 1 / (2 * sigma_square) * distance)
        
        assert density.shape == torch.Size(np.array([B, 1, H, W]))
        
        return density


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(in_channels),
            # nn.ReLU6(inplace=True),
            nn.Sigmoid()           
        )
        
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU6(inplace=True),
            nn.Sigmoid()            
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU6(inplace=True),
            nn.Sigmoid()
            # nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True),                      
        )                        
        
        self.relu = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True),        
            # nn.ReLU6(inplace=True)
            nn.Sigmoid()
        )
            
    
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.downsample(x)
        assert x.shape == torch.Size(np.array([B, C, int(H/2), int(W/2)]))
                
        x = self.conv1(x)
        assert x.shape == torch.Size(np.array([B, 2 * C, int(H/2), int(W/2)]))
        
        x = self.conv2(x) + x
        # x = self.conv2(x)
        assert x.shape == torch.Size(np.array([B, 2 * C, int(H/2), int(W/2)]))
        
        x = self.relu(x)
        
        return x
        

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2, bias=True),            
            nn.BatchNorm2d(out_channels),
            # nn.ReLU6(inplace=True),
            nn.Sigmoid()            
        )
        
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU6(inplace=True),
            nn.Sigmoid()            
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),            
            # nn.ReLU6(inplace=True),           
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),        
            nn.Sigmoid()
        )
                   
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.upsample(x)
        x = self.conv2(x) + x
        # x = self.conv2(x)
        x = self.sigmoid(x)
        
        return x