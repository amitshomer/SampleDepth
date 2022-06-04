from torch import nn
import torch
from torchsummary import summary
import numpy as np


class Global_mask(nn.Module):
    def __init__(self, batch_size):
        super().__init__()

        self._global_mask = torch.nn.Parameter(torch.tensor(torch.ones(batch_size, 1, 256, 1216)))

    def forward(self, input: torch.tensor):
        
        sample_out = self._global_mask * input

        return sample_out, None , self._global_mask

    

# if __name__ == '__main__':
#     # model = SampleDepth(in_channels=1,
#     #          out_channels=2,
#     #          n_blocks=4,
#     #          start_filters=32,
#     #          activation='relu',
#     #          normalization='batch',
#     #          conv_mode='same',
#     #          n_sample = 10000,
#     #          dim=2)

#     # x = torch.randn(size=(2, 1, 256, 1216), dtype=torch.float32)
#     # with torch.no_grad():
#     #     sample_points, bin_mask = model(x)
    
#     # l1= model.sample_number_loss(bin_mask)

#     # # print(f'Out: {out.shape}')
#     # # shape = 1216*256
#     # # out = compute_max_depth(shape, print_out=True, max_depth=10)


