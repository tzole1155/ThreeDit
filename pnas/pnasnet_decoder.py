from __future__ import print_function, division, absolute_import
from collections import OrderedDict

import toolz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import logging

log = logging.getLogger(__name__)



class Up(nn.Sequential):
    def __init__(self, num_input_channels, num_output_channels):
        super(Up, self).__init__()
        self.convA = nn.Conv2d(num_input_channels, num_output_channels, kernel_size=3, stride=1, padding=1)
        self.convB = nn.Conv2d(num_output_channels, num_output_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_up = F.interpolate(x, size=[x.size(2)*2, x.size(3)*2], mode='bilinear', align_corners=True)
        x_convA = self.relu(self.convA(x_up))
        x_convB = self.relu(self.convB(x_convA))
        return x_convB

class PNASNet5LargeDecoder(nn.Module):
    def __init__(self, 
        decoder_scale:  int=1024,
        freeze:             bool=False,
    ):
        super(PNASNet5LargeDecoder, self).__init__()

        num_channels_d32_in = 4320
        num_channels_d32_out = decoder_scale

        self.freeze = freeze

        self.conv_d32 = nn.Conv2d(num_channels_d32_in, num_channels_d32_out, kernel_size=1, stride=1)

        self.up1 = Up(num_input_channels=num_channels_d32_out // 1, num_output_channels=num_channels_d32_out // 2)
        self.up2 = Up(num_input_channels=num_channels_d32_out // 2, num_output_channels=num_channels_d32_out // 4)
        self.up3 = Up(num_input_channels=num_channels_d32_out // 4, num_output_channels=num_channels_d32_out // 8)
        self.up4 = Up(num_input_channels=num_channels_d32_out // 8, num_output_channels=num_channels_d32_out // 16)
        self.up5 = Up(num_input_channels=num_channels_d32_out // 16, num_output_channels=num_channels_d32_out // 32)
        self.conv3 = nn.Conv2d(num_channels_d32_out // 32, 1, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU()

        # if ckpt:
        #     stored = torch.load(ckpt)
        #     self.load_state_dict(toolz.keymap(lambda k: k.replace('module.', ''), stored))

    def forward_train(self,
        features: torch.Tensor
    ) -> torch.Tensor:
        
        decoder_in = self.conv_d32(features)

        decoder_x2 = self.up1(decoder_in)
        decoder_x4 = self.up2(decoder_x2)
        decoder_x8 = self.up3(decoder_x4)
        decoder_x16 = self.up4(decoder_x8)
        decoder_x32 = self.up5(decoder_x16)
        output = self.conv3(decoder_x32)

        output = self.activation(output)

        return output
    
    def forward_eval(self,
        features: torch.Tensor
    )-> torch.Tensor:
        with torch.no_grad():
            decoder_in = self.conv_d32(features)
            decoder_x2 = self.up1(decoder_in)
            decoder_x4 = self.up2(decoder_x2)
            decoder_x8 = self.up3(decoder_x4)
            decoder_x16 = self.up4(decoder_x8)
            decoder_x32 = self.up5(decoder_x16)
            output = self.conv3(decoder_x32)

            output = self.activation(output)

            return output
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        
        if self.freeze and self.training:
            log.warning("Pnas decoder is frozen but in training state, reverting to eval state.")
            self.eval()
        
        return self.forward_eval(features) if self.freeze else self.forward_train(features)