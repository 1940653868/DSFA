import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from linear_attention_transformer import LinearAttentionTransformer
try:
    from cwt_block import CwtPatchEmbed
except:
    from model.cwt_block import CwtPatchEmbed


def conv1x3(in_planes, out_planes, kernal_width=5, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, kernal_width), stride=(1, stride),
                     padding= (0,int((kernal_width-1)/2)), bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.GELU()
        self.conv2 = conv1x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv1xN(in_planes, out_planes,kernal_with=10, stride=10):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=(1, kernal_with), stride=(1, stride),
                     padding=1, bias=False)

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1xN(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.GELU(inplace=True)
        self.conv2 = conv1xN(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


from collections import OrderedDict
class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, conv_bias, ada_poolsize = 64, feadim = 256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inchannel = in_channels
        self.out_channels = 2 ** (math.floor(np.log2(in_channels)) + 1)
        out_channels = 2 ** (math.floor(np.log2(in_channels)) + 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 7), stride = (1, 2), 
                        padding=(0, 3), bias= conv_bias),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))
        )
        
        # self.conv2 = BasicBlock(inplanes=out_channels, planes=int(2*out_channels), stride=1, )
        self.conv2 = BasicBlock(inplanes=out_channels, planes=int(1*out_channels), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))
        self.conv3 = BasicBlock(inplanes=int(1*out_channels), planes=int(1*out_channels), stride=1)
        self.pool3 = nn.AdaptiveAvgPool2d((1, ada_poolsize))
        self.drop = nn.Dropout(p=0.25)
        self.project = nn.Linear(in_features=ada_poolsize, out_features = feadim)
    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        x = self.pool2(x)
        print(x.shape)
        x = self.conv3(x)
        x = self.pool3(x)
        print(x.shape)
        x = self.drop(x)
        x = self.project(x)
        return x


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101, enabale_deeper=False):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)
        self.enabale_deeper = enabale_deeper
        self.gelu = nn.GELU()
        self.projection2 = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        if self.enabale_deeper:
            x = self.gelu(x)
            x = self.projection2(x)

        return x


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class StyliezeBIOTEncoder(nn.Module):
    def __init__(
        self,
        emb_size=256,
        heads=8,
        depth=4,
        n_channels=16,
        n_fft=200,
        hop_length=100,
        **kwargs
    ):
        super().__init__()
        if 'args' in kwargs:
            print('load args')
            self.args = kwargs['args']
        else:
            self.args = None

        # selcet the embedding method
        # STFT(time,freq,both),MLP,Conv
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.merge_group_mode = "concat" #["add", "concat"] 
        self.enable_deep_patch = True
        self.level_num = self.args.dwt_level+1 
        self.emb_size = emb_size
        self.style_dwtlevel = self.args.style_dwtlevel 
        if self.args.encdata_mode in ['dwt']:
            
            if self.merge_group_mode =='concat':
                # Ensure that concat_embsize is an integer and even, in order to serve position encoding.
                concat_embsize = int(emb_size/self.level_num) if (int(emb_size/self.level_num)%2 == 0) else int(emb_size/self.level_num-1)
                self.group_patch_embedding = nn.ModuleList(
                [PatchFrequencyEmbedding(emb_size= concat_embsize, n_freq=self.n_fft // 2 + 1, enabale_deeper=self.enable_deep_patch) for _ in range(self.level_num)]
            )
            elif self.merge_group_mode == 'add':
                self.group_patch_embedding = nn.ModuleList(
                [PatchFrequencyEmbedding(emb_size=emb_size, n_freq=self.n_fft // 2 + 1, enabale_deeper=self.enable_deep_patch) for _ in range(2)]
            )
            self.AddMergeEmbedding =nn.Sequential( 
                nn.GELU(),
                nn.Linear(in_features=emb_size, out_features= emb_size)
            )
        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )

        # self.cwt_embedding = CwtPatchEmbed(img_size=())
        hz_size = 32
        self.cwt_projection = CwtPatchEmbed(img_size=(hz_size, 2000), patch_size=(hz_size, 50), stride=(hz_size, 50), 
                                        in_chans=n_channels, embed_dim_ratio=16, emb_size=emb_size)
    
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,     # dropout right aPositionalEncodingfter self-attention layer
            attn_dropout=0.2,           # dropout post-attention
        )
        

        # channel token, N_channels >= your actual channels
        if self.merge_group_mode == 'concat' and self.args.encdata_mode in ['dwt']:
            self.channel_tokens = nn.Embedding(n_channels, concat_embsize)
            self.positional_encoding = PositionalEncoding(concat_embsize)
        else:
            self.channel_tokens = nn.Embedding(n_channels, emb_size)
            self.positional_encoding = PositionalEncoding(emb_size)


        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

        # stylize paraments
        self.enable_stylize = True
        if 'args' in kwargs:
            self.scaling_factor = self.args.scaling_factor 
        else:
            print('without seting args, set default scaling_factor is 15')
            self.scaling_factor = 15  

        self.drop_HorL = "no"  #["H","L","no"]
        self.restyle_dim = "time"  #time, channel
        print( "drop_HorL",self.drop_HorL, "restyle_dim", self.restyle_dim, self.enable_stylize, self.merge_group_mode, 'level+1',self.level_num )

    def stft(self, sample):
        spectral = torch.stft( 
            input = sample.squeeze(1),
            n_fft = self.n_fft, 
            hop_length = self.hop_length,
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)
    def padding_freqemb(self, concat_tensor, emb_size):
        padding_needed = emb_size - concat_tensor.size(-1)
        padded_tensor = F.pad(concat_tensor, (0, padding_needed))

        
        return padded_tensor
    def stylize_feature(self, LL):
        """
        augment feature
        """
        B, C, H, W = LL.shape
        LL_cp = LL.view(C, -1)  # MEAN / STD : (C) -> batch-wise mean of each-channel

        # Calculate batch-wise statistics
        mean_LL = torch.mean(LL_cp, dim=1)   # batch-wise mean
        std_LL = torch.std(LL_cp, dim=1)    # batch-wise std
        # print(mean_LL.shape)
        # Calculate channel-wise statistics of mean and std vector
        mu_hat_LL = mean_LL.mean()
        sigma_hat_LL = mean_LL.std()
        # print(mu_hat_LL, sigma_hat_LL)

        mu_tilde_LL = std_LL.mean()
        sigma_tilde_LL = std_LL.std()

        # Sample new style vectors from the manipulated distribution
        mu_new = torch.normal(mu_hat_LL.view(1, 1).repeat(B, C), self.scaling_factor * sigma_hat_LL.view(1, 1).repeat(B, C)) #output : (B, C)
        sigma_new = torch.normal(mu_tilde_LL.view(1, 1).repeat(B, C), self.scaling_factor * sigma_tilde_LL.view(1,1).repeat(B, C))

        mu_new_reshape = mu_new.view(B, C, 1, 1).repeat(1, 1, H, W)
        sigma_new_reshpae = sigma_new.view(B, C, 1, 1).repeat(1, 1, H, W)

        # Equation 6 ~ Normalize original feature with batch statistics
        normalized_LL = (LL - mean_LL.view(1, C, 1, 1).repeat(B, 1, H, W)) /( std_LL.view(1, C, 1, 1).repeat(B, 1, H, W) + 1e-6)
        # Equation 6 ~ Affine transformation with sampled style vectors
        stylized_LL = sigma_new_reshpae * normalized_LL + mu_new_reshape

        return stylized_LL
    
    def Dwtembedding_forward(self, x, n_channel_offset=0, perturb=False, enable_stylize = False):
        Groupsignal = x 
        batch, signalgroup, channel, step = Groupsignal.shape
        # stft mode
        Group_emb = []
        style_dwtlevel_list =  [signalgroup - item - 1  for item in range(self.args.dwt_level)]
        for index in range(signalgroup):
            x = Groupsignal[:, index] #[batch, group, channel, timestep]
            emb_seq = []
            for i in range(channel): # x.shape torch.Size([64, 16, 2000])
                channel_spec_emb = self.stft(x[:, i : i + 1, :]) # channel_spec_emb.shape torch.Size([64, 101, 19]) 19time-seq 101-freq_size

                channel_spec_emb = self.group_patch_embedding[index](channel_spec_emb) # channel_spec_emb torch.Size([64, 19, 256])
                # if enable_stylize and index == (signalgroup-1): #【最高频增强】
                if enable_stylize and (index in style_dwtlevel_list): 
                    channel_spec_emb = torch.unsqueeze(channel_spec_emb, dim=2) # [batch,  time_seq, 1, embedding]
                    channel_spec_emb = self.stylize_feature(channel_spec_emb)  # input should like: [batch, channel, w, h]
                    channel_spec_emb = torch.squeeze(channel_spec_emb, dim=2)

                batch_size, ts, _ = channel_spec_emb.shape
                # (batch_size, ts, emb)
                channel_token_emb = (
                    self.channel_tokens(self.index[i + n_channel_offset])
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(batch_size, ts, 1)
                )
                
                channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)

                emb_seq.append(channel_emb) 
            emb_seq = torch.cat(emb_seq, dim=1) 
            Group_emb.append(emb_seq)
        
        if self.merge_group_mode =='add':
            sum_emb = torch.sum(torch.stack(Group_emb, dim=0), dim=0)
        elif self.merge_group_mode == 'concat':
            sum_emb = torch.concat(Group_emb, dim=-1)  #[256/n]*n
            if self.emb_size - sum_emb.size(-1)>0:
                sum_emb = self.padding_freqemb(sum_emb, emb_size=self.emb_size)      
        else:
            raise f'not implement merge choice: {self.merge_group_mode}'
        
        # perturb -- random mask when pretrain model
        if perturb:
            ts = sum_emb.shape[1]
            ts_new = np.random.randint(ts // 2, ts)
            selected_ts = np.random.choice(range(ts), ts_new, replace=False)
            sum_emb = sum_emb[:, selected_ts]        

        emb = sum_emb

        return emb      
    def Dwtembedding_StyleinChanel_forward(self, x, n_channel_offset=0, perturb=False, enable_stylize = False):
        # restyle in channel必须等待channel维度的特征concat之后再进行操作所以对比原来的函数改动会比较大，因此重新实现，
        # 效果不佳，原因是重采样的过程中，只有一个系数代表了所有的通道，忽略了通道之间的差异性
        # 整体上代码可能比较臃肿
        # for loop in level
            #  for loop in channel
            #  concat channel
            #  restyle in channel
        # concat in level
        
        Groupsignal = x 
        batch, signalgroup, channel, step = Groupsignal.shape
        # stft mode
        Group_emb = []
        style_dwtlevel_list =  [signalgroup - item-1  for item in range(self.args.dwt_level)]
        for index in range(signalgroup):
            x = Groupsignal[:, index] #[batch, group, channel,timestep]
            # [approx, detail] index 越大,成分越接近细节
            emb_seq = []
            for i in range(channel): # x.shape torch.Size([64, 16, 2000])
                channel_spec_emb = self.stft(x[:, i : i + 1, :]) # channel_spec_emb.shape torch.Size([64, 101, 19]) 19time-seq 101-freq_size

                # print('before mlp',channel_spec_emb.mean())
                channel_spec_emb = self.group_patch_embedding[index](channel_spec_emb) # channel_spec_emb torch.Size([64, 19, 256])
                
                # print('after mlp',channel_spec_emb.mean())
                # print('weight', self.group_patch_embedding[index].projection.weight.mean())
                # if enable_stylize and index == (signalgroup-1):
                # if enable_stylize and (index in style_dwtlevel_list):
                # # if enable_stylize and index == 0:                
                #     # 对齐维度的设计上有两个切入思路
                #     # A 【batch，timeblock,1 ,fea】 对于每一个timeblock其实是对其channel，互相语言可以转换的
                #     # B 【batch， channel， timeblock， fea】 需要修改的代码，将 pos embedding转移位置
                #     # Waring: 通道的数目比较少的时候，计算分布没有实际的意义，可能有反向的效果；比如目前的timeblock只有9大小比较尴尬，作者
                #     # 用在Resnet结构上，channel的size可以达到64-512
                #     channel_spec_emb = torch.unsqueeze(channel_spec_emb, dim=2)
                #     channel_spec_emb = self.stylize_feature(channel_spec_emb)  # input should like: [batch, channel, w, h]
                #     channel_spec_emb = torch.squeeze(channel_spec_emb, dim=2)
                #添加If enable_stylize可以约束是否在两个分支zero
                if enable_stylize:
                # if True:
                    if self.drop_HorL == "H" and index == (signalgroup-1):
                        # print('zero in H')
                        channel_spec_emb = torch.zeros_like(channel_spec_emb)
                    elif self.drop_HorL == "L"and index == 0:
                        # print('zero in L')
                        channel_spec_emb = torch.zeros_like(channel_spec_emb)
                    else:
                        pass
                batch_size, ts, _ = channel_spec_emb.shape
                init_feature = channel_spec_emb
                # (batch_size, ts, emb)
                emb_seq.append(init_feature) 
            # 组合16通道的成分
            # emb_seq = torch.cat(emb_seq, dim=1) stack[channel, batch, seq, dim]
            Allchannel_seq = torch.stack(emb_seq).permute(1,0,2,3).contiguous()  # [batch, channel , time_seq, dim]
            style_channel_seq = self.stylize_feature(Allchannel_seq)  #内部只有[channel]个分布，而过去的方法内部有[channel*time_seq]个分布？
            # 
            style_emb_seq = []               
            for i in range(channel):     
                channel_spec_emb = style_channel_seq[:,i]
                channel_token_emb = ( #input->torch.Size([64, 19, 256])
                    self.channel_tokens(self.index[i + n_channel_offset])
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(batch_size, ts, 1)
                )
                # (batch_size, ts, emb)
                channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)
                

                style_emb_seq.append(channel_emb) 
            # 组合16通道的成分
            emb_seq = torch.cat(style_emb_seq, dim=1) 
            


            # 将两个dwt成分加入Group_emb
            Group_emb.append(emb_seq)
        # 如何组合两个成分
        if self.merge_group_mode =='add':
            # add
            sum_emb = torch.sum(torch.stack(Group_emb, dim=0), dim=0)
            
        elif self.merge_group_mode == 'concat':# concat
            # 理论上应该先完成concat, 再进入随机select ts block的环节
            # for item in Group_emb:
            #     print(item.shape)
            sum_emb = torch.concat(Group_emb, dim=-1)  #[256/n]*n
            if self.emb_size - sum_emb.size(-1)>0:
                sum_emb = self.padding_freqemb(sum_emb, emb_size=self.emb_size)
            
        else:
            raise f'not implement merge choice: {self.merge_group_mode}'
        # perturb
        if perturb:
            ts = sum_emb.shape[1]
            ts_new = np.random.randint(ts // 2, ts)
            selected_ts = np.random.choice(range(ts), ts_new, replace=False)
            # 原维度的所有可能特征位置中无放回地随机选择 ts_new 个特征索引(),
            # 相当于ts维度做了mask
            sum_emb = sum_emb[:, selected_ts]        

        emb = sum_emb

        return emb      

    def forward(self, x, n_channel_offset=0, perturb=False):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
        restyle_dim = self.restyle_dim
        emb_seq = []
        if self.args.encdata_mode in ['dwt']:
            emb         = self.Dwtembedding_forward(x=x, n_channel_offset=0, perturb=perturb, enable_stylize=False)
            if restyle_dim == 'time':
                restyle_emb = self.Dwtembedding_forward(x=x, n_channel_offset=0, perturb=perturb, enable_stylize=True)            
            elif restyle_dim == "channel":
                restyle_emb = self.Dwtembedding_StyleinChanel_forward(x=x, n_channel_offset=0, perturb=perturb, enable_stylize=True) 
            else:
                raise f'not implement restyle_dim {restyle_dim}'
            # emb = self.AddMergeEmbedding(emb)
            # restyle_emb = self.AddMergeEmbedding(restyle_emb)
        elif self.args.encdata_mode in ['time_only','freq_only','both']: #choices=['time_only','freq_only','both','cwt','dwt'])
            # stft mode
            for i in range(x.shape[1]): # x.shape torch.Size([64, 16, 2000])
                channel_spec_emb = self.stft(x[:, i : i + 1, :]) # channel_spec_emb.shape torch.Size([64, 101, 19]) 19time-seq 101-freq_size
                channel_spec_emb = self.patch_embedding(channel_spec_emb) # channel_spec_emb torch.Size([64, 19, 256])
                batch_size, ts, _ = channel_spec_emb.shape
                # (batch_size, ts, emb)
                channel_token_emb = ( #input->torch.Size([64, 19, 256])
                    self.channel_tokens(self.index[i + n_channel_offset])
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(batch_size, ts, 1)
                )
                # (batch_size, ts, emb)
                channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)

                # perturb
                if perturb:
                    ts = channel_emb.shape[1]
                    ts_new = np.random.randint(ts // 2, ts)
                    selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                    channel_emb = channel_emb[:, selected_ts]
                emb_seq.append(channel_emb) 
            emb = torch.cat(emb_seq, dim=1) #concat channels
            restyle_emb = emb
        else:
            raise NotImplementedError

        # (batch_size, emb)
        emb = self.transformer(emb).mean(dim=1)
        restyle_emb = self.transformer(restyle_emb).mean(dim=1)
        return emb, restyle_emb


# supervised classifier module
class  StyliezeBIOTClassifier(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_classes=6, **kwargs):
        super().__init__()
        self.biot = StyliezeBIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x, x_tr = self.biot(x)
        y = self.classifier(x)
        y_tr = self.classifier(x_tr)
        return (y, x), (y_tr, x_tr)


# Stylizeunsupervised pre-train module
class  StyliezeUnsupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_channels=18, **kwargs):
        super(StyliezeUnsupervisedPretrain, self).__init__()
        self.biot = StyliezeBIOTEncoder(emb_size, heads, depth, n_channels, **kwargs)        
        self.prediction_mask = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.prediction_trans = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
    def forward(self, x, n_channel_offset=0):
        # emb = self.biot(x, n_channel_offset, perturb=True)
        Maskemb, Maskemb_trans = self.biot(x, n_channel_offset, perturb=True)
        Maskemb = self.prediction_mask(Maskemb)
        pred_emb, emb_trans = self.biot(x, n_channel_offset) # pred_emb 用于分类的真实emb
        emb_trans = self.prediction_trans(emb_trans)
        return Maskemb, emb_trans, pred_emb


# supervised pre-train module
class  StyliezeSupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, **kwargs):
        super().__init__()
        if 'args' in kwargs:
            self.args = kwargs['args']
        else:
            self.args = None
        # self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, args = self.args)
        self.biot = StyliezeBIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, args = self.args)
        self.classifier_chb_mit = ClassificationHead(emb_size, 1)
        self.classifier_iiic_seizure = ClassificationHead(emb_size, 6)
        self.classifier_tuab = ClassificationHead(emb_size, 1)
        self.classifier_tuev = ClassificationHead(emb_size, 6)

    def forward(self, x, task="chb-mit"):
        x, x_trans = self.biot(x)
        if task == "chb-mit":
            x = self.classifier_chb_mit(x)
            x_trans = self.classifier_chb_mit(x_trans)
        elif task == "iiic-seizure":
            x = self.classifier_iiic_seizure(x)
            x_trans = self.classifier_iiic_seizure(x_trans)
        elif task == "tuab":
            x = self.classifier_tuab(x)
            x_trans = self.classifier_tuab(x_trans)
        elif task == "tuev":
            x = self.classifier_tuev(x)
            x_trans = self.classifier_tuev(x_trans)
        else:
            raise NotImplementedError
        return x, x_trans


if __name__ == "__main__":
    x = torch.randn(2, 16, 2, 2000)
    # model = StyliezeBIOTClassifier(n_fft=200, hop_length=200, depth=4, heads=8)
    # out = model(x)
    # print(out.shape)

    # model = StyliezeUnsupervisedPretrain(n_fft=200, hop_length=200, depth=4, heads=8)
    # out1, out2 = model(x)
    # print(out1.shape, out2.shape)
    x = torch.randn(2, 16, 2000)
    x = torch.unsqueeze(x, dim=2)
    model = ConvEmbedding(in_channels=16,conv_bias=False, ada_poolsize=128, feadim=256)
    y = model(x)
    print(y.shape)
