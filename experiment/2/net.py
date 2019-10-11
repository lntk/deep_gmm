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
            
            """
            EXP 2
            """
            # pi, mu, sigma, output = self.gmm_block(curr_frame, pi, mu, sigma)            
            
            """
            EXP 3
            """
            output = self.gmm_block(curr_frame)
            
            outputs.append(output)
                                                                   
        outputs = torch.stack(outputs, dim=1)                
        
        return outputs
    
    def init_gmm_from_first_frame(self, frame):
        B, C, H, W = frame.shape
        K = self.num_component
        
        assert K != 1
        
        pi = torch.ones(B, C * K, H, W) * (0.1 / (K - 1))
        pi[:, :K, :, :, :] = torch.ones(B, C, H, W) * 0.9 
        
        mu = torch.zeros(B, C * K, H, W)  # B x (C * K) X H x W 
        mu[:, :K, :, :, :] = frame
        
        sigma = torch.ones(B, C * K, H, W)  # B x (C * K) X H x W 
        
        return pi, mu, sigma
    
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
        
        """
        EXP 2
        """        
        # self.f_alpha = self.nonlinear_f(layers=[in_channels + num_component * (2 + in_channels), 16, 8, 4, num_component])
        
        
        # self.f_pi    = self.nonlinear_f(layers=[2 * num_component, 16, 8, 4, num_component], last_activation="softmax")        
        # self.f_mu    = self.nonlinear_f(layers=[in_channels + in_channels * num_component + num_component, 16, 8, 8, in_channels * num_component])
        # self.f_sigma = self.nonlinear_f(layers=[in_channels + in_channels * num_component + num_component, 16, 8, 4, num_component])
        
        """
        EXP 2
        """
        # self.f_tau   = self.nonlinear_f(layers=[num_component, 16, 8, 4, 1])
        # self.f_gamma = GammaReLU.apply
        
        
        """
        EXP 2.2
        """
        # self.f_gamma = self.nonlinear_f(layers=[num_component, 16, 8, 4, 1])        
        
    
    def forward(self, x, pi, mu, sigma):
        B, C, H, W = x.shape
        K = self.num_component        
        channel_dim = 1
        
        assert pi.shape == torch.Size(np.array([B, K, H, W]))
        assert mu.shape == torch.Size(np.array([B, C * K, H, W]))
        assert sigma.shape == torch.Size(np.array([B, K, H, W]))        
        
        """
        EXP 2
        """
        # alpha = self.f_alpha(torch.cat([x, pi, mu, sigma], dim=channel_dim))        
        # assert alpha.shape == torch.Size(np.array([B, K, H, W]))    
        
        """
        EXP 2.2
        """
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
        
        # tau = self.f_tau(pi)
        # assert tau.shape == torch.Size(np.array([B, 1, H, W]))
        
        # beta = torch.log(1 + torch.exp(config.SCALING_FACTOR * (pi - tau))) / config.SCALING_FACTOR + tau
        # assert beta.shape == torch.Size(np.array([B, K, H, W]))
        
        # chi = beta * torch.cat([self.multivariate_Gaussian(x, mu[:, i * C: (i + 1) * C, :, :], sigma[:, i: i+1, :, :]) for i in range(self.num_component)], dim=channel_dim)
        # chi = torch.sum(chi, dim=channel_dim, keepdim=True)
        # assert chi.shape == torch.Size(np.array([B, 1, H, W]))
        
        
        # gamma = self.f_gamma(chi)
        # gamma = 1 - gamma
        # assert gamma.shape == torch.Size(np.array([B, 1, H, W]))
        
        """
        EXP 2.2
        """
        gamma = self.f_gamma(pi * torch.cat([self.multivariate_Gaussian(x, mu[:, i * C: (i + 1) * C, :, :], sigma[:, i: i+1, :, :]) for i in range(self.num_component)], dim=channel_dim))        
        
        return pi, mu, sigma, gamma
        
    
    def nonlinear_f(self, layers, last_activation="relu"):
        in_channels = layers[0]
        out_channels = layers[-1]
        
        modules = [
            # nn.BatchNorm2d(in_channels)
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
    
    
class GammaReLU(torch.autograd.Function):        
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)      
        output = input.clone()
        output[output < config.FOREGROUND_THRESHOLD] = 0
        
        return output
        # return input.clamp(min=config.FOREGROUND_THRESHOLD)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < config.FOREGROUND_THRESHOLD] = 0
        return grad_input