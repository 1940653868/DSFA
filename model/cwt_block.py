import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
def to_2tuple(input_value):
    if isinstance(input_value, (list, tuple)):
        if len(input_value) == 2:
            return tuple(input_value)
        elif len(input_value) == 1:
            input_value = int(input_value[0])
            return  (input_value, input_value)
        else:
            raise ValueError("Input list/tuple must have exactly two elements.")
            
        
    elif isinstance(input_value, (int, float)):
        return (input_value, input_value)
    else:
        raise TypeError("Unsupported input type.")

class CwtPatchEmbed(nn.Module):
    """ eeg signal  to Patch Embedding
    """
    def __init__(self, img_size=(100, 2000), patch_size=(100, 200), stride =(100, 100), in_chans=16, embed_dim_ratio=32, emb_size = 256):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim_ratio = embed_dim_ratio
        embed_dim = in_chans*embed_dim_ratio
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, groups=in_chans, bias= False)
        self.cwt_2_dim = nn.Linear(embed_dim_ratio, emb_size)
        # Depthwise Convolution分组卷积操作) #
        self.instan_nrom = nn.InstanceNorm2d(num_features=in_chans, affine=True)
    def forward(self, x):
        # input:batch,channel_fea,time_step,hz(1)
        x = x.transpose(-1, -2)
        B, C, Hz, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2) #batch,channel_fea,hz(1),time_step
        x = x.view(B, self.embed_dim_ratio, C, -1) #分离出通道与整合特征，卷积之中每个通道被扩展为embed_dim_ratio倍
        x = torch.permute(x, (0, 2, 3, 1))
        # x = self.instan_nrom(x)
        x = self.cwt_2_dim(x) #shape[batch,channel,time_blcok, feature]
        
        return x

if __name__ =="__main__":
    # 示例用法
    width = 32
    input_list = [64]
    input_tuple = (64,)

    output_width = to_2tuple(width)  # 输出：(32, 32)
    output_list = to_2tuple(input_list)  # 输出：(64, 64)
    output_tuple = to_2tuple(input_tuple)  # 输出：(64, 64)
    print(output_width, output_list, output_tuple)

    model = CwtPatchEmbed(img_size=(100, 2000), patch_size=(100, 200), stride =(100, 100), in_chans=16, embed_dim_ratio=32)
    input_data = torch.zeros((7, 16, 2000, 100))
    y = model(input_data)
    print(y.shape)