from torch import nn
import torch
import config
from layer.encoder import Encoder


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, 
                 num_component, 
                 kernel_size, 
                 stride=1, 
                 padding=1, 
                 input_size=config.INPUT_SIZE, 
                 sequence_length=config.SEQUENCE_LENGTH):
        super().__init__()
        
        # hidden_channels = num_component * 3  # Gaussian
        hidden_channels = num_component * 2  # Bernoulli
        encoder_out_channels = 16
        
        self.encoder = Encoder(in_channels=in_channels,
                               out_channels=encoder_out_channels)

        self.W_xi = nn.Conv2d(in_channels=encoder_out_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.W_hi = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.W_xf = nn.Conv2d(in_channels=encoder_out_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.W_hf = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)        
        self.W_xc = nn.Conv2d(in_channels=encoder_out_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.W_hc = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.W_xo = nn.Conv2d(in_channels=encoder_out_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.W_ho = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        self.W_ci = nn.Parameter(torch.zeros(hidden_channels, input_size[0], input_size[1]))
        self.W_cf = nn.Parameter(torch.zeros(hidden_channels, input_size[0], input_size[1]))
        self.W_co = nn.Parameter(torch.zeros(hidden_channels, input_size[0], input_size[1]))

        self.hidden_channels = hidden_channels
        self.input_size = input_size
        self.sequence_length = sequence_length

    def forward(self, x):
        """
        
        :param x: batch_size x sequence_length x channel x height x width
        :return:
        """

        outputs = list()
        batch_size = x.shape[0]
        
        h, c = self.init_hidden(hidden_shape=(batch_size, self.hidden_channels, self.input_size[0], self.input_size[1]),
                                device=x.get_device())
        
        for i in range(self.sequence_length):
            curr_x = x[:, i, :, :, :]           
            
            # Encode
            curr_x = self.encoder(curr_x)
            
            # ConvLSTM        
            c_temp = torch.sigmoid(self.W_xi(curr_x) + self.W_hi(h) + self.W_ci * c)
            remember = torch.sigmoid(self.W_xf(curr_x) + self.W_hf(h) + self.W_cf * c)
            save = torch.tanh(self.W_xc(curr_x) + self.W_hc(h))
            c = remember * c + save * c_temp
            focus = torch.sigmoid(self.W_xo(curr_x) + self.W_ho(h) + self.W_co * c)
            h = focus * torch.tanh(c)
            outputs.append(h)
            
        outputs = torch.stack(outputs, dim=1)                
        
        return outputs

    @staticmethod
    def init_hidden(hidden_shape, device):
        h = torch.rand(hidden_shape).to(device)
        c = torch.rand(hidden_shape).to(device)
        return h, c
