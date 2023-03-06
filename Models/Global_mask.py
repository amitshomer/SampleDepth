from torch import nn
import torch
import numpy as np
from .SampleDepth import Softargmax2d

class Global_mask(nn.Module):
    def __init__(self, batch_size, multi, dataset):
        super().__init__()
        # if multi:
        #     first_layer = torch.zeros(int(batch_size/2.0), 1, 256, 1216)
        #     sec_layer = torch.ones(int(batch_size/2.0), 1, 256, 1216)
        # else:
        #     first_layer = torch.zeros((torch.ones(int(batch_size), 1, 256, 1216)))
        #     sec_layer = torch.ones(int(batch_size), 1, 256, 1216)

        # self._global_mask = torch.nn.Parameter(torch.cat(( torch.rand(int(batch_size/2.0), 1, 256, 1216), torch.rand(int(batch_size/2.0), 1, 256, 1216)), dim = 1))
        if dataset =="SHIFT":
            H=400
            W=640
        else: # Kitti dataset
            H=256
            W=1216

        if multi and batch_size>1:
            w = torch.empty(int(batch_size/2),1,H, W)
            u= torch.empty(int(batch_size/2),1,H, W)
        else:
            w = torch.empty(batch_size,1,H, W)
            u= torch.empty(batch_size,1,H, W)



        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(u, mode='fan_out', nonlinearity='relu')
        self._global_mask = torch.nn.Parameter(torch.cat((w,u),dim=1))
    
        self.relu = torch.nn.ReLU(inplace=False)
        
        self.soft_max = nn.Softmax(dim=1)
        
        self.softargmax = Softargmax2d(initial_temperature = 0.0001)
    
    def forward(self, sampler_from: torch.tensor):
        global_maks_relu = self.softargmax(self.soft_max(self._global_mask)).unsqueeze(dim=1)
        sample_out = global_maks_relu * sampler_from

        return sample_out, None , global_maks_relu

    # def global_mask_loss(self ):
    #     mask_half = torch.ones(self._global_mask.shape).cuda() *0.5
        
    #     out = torch.sum(torch.abs(torch.abs(self._global_mask - mask_half) - mask_half))
       
    #     return out

    
    def sample_loss(self, pred_map, n_sample ):
        batch_size = pred_map.size(0)
        return torch.abs((pred_map.sum()/batch_size)-n_sample)/n_sample

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


