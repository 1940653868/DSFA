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
class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
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


class BIOTEncoder(nn.Module):
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

        self.n_fft = n_fft
        self.hop_length = hop_length

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
        self.positional_encoding = PositionalEncoding(emb_size)

        # channel token, N_channels >= your actual channels
        self.channel_tokens = nn.Embedding(n_channels, 256)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

    def stft(self, sample):
        spectral = torch.stft( 
            input = sample.squeeze(1),
            n_fft = self.n_fft, #FFT的大小决定了频率分辨率,较大的n_fft能够获得更好的频率分辨率,但也会增加计算量 ; 返回的频率范围为nfft 如果oneside=True，为1/2 n_fft
            hop_length = self.hop_length,#这个参数指定了STFT中相邻窗口之间的步长(以样本为单位)。较小的hop_length会产生更多的重叠,从而获得更好的时间分辨率
            center = False,
            onesided = True,
            return_complex = True,
        )
        # num_frames 取决于信号长度，win_len和 hop_length。
        # num_fft_bins 通常为 n_fft // 2 + 1,因为只返回单边频谱。
        # return [batch, N, T] N is the number of frequencies where STFT is applied and T is the total number of frames used.
        # return_complex = True,  (batch, N,T, 2)
        return torch.abs(spectral)

    def forward(self, x, n_channel_offset=0, perturb=False):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
        emb_seq = []
        if self.args.encdata_mode in ['time_only','freq_only','both', 'dwt']: #choices=['time_only','freq_only','both','cwt','dwt'])
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
                    # 原维度的所有可能特征位置中无放回地随机选择 ts_new 个特征索引(),
                    # 相当于ts维度做了mask
                    # print(ts_new)
                    channel_emb = channel_emb[:, selected_ts]
                emb_seq.append(channel_emb) 

        # cwt mode
        elif self.args.encdata_mode == 'cwt':
            cwt_emb = self.cwt_projection(x)    #input shape[batch, channel, timestep, hz] output_shape:[batch, channel, time_blcok, feature]
            # channel_embedding and seq_embedding
            batch_size, channel_size, ts, _ = cwt_emb.shape
            # (batch_size, ts, emb)
            for i in range(channel_size):
                channel_token_emb = ( #input->torch.Size([64, 19, 256])
                    self.channel_tokens(self.index[i + n_channel_offset])
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(batch_size, ts, 1)
                )  
                channel_emb = self.positional_encoding(cwt_emb[:,i] + channel_token_emb)
                emb_seq.append(channel_emb) 
        
        # elif 
        # (batch_size, 16 * ts, emb)
        
        emb = torch.cat(emb_seq, dim=1) #concat channels
        # (batch_size, emb)
        emb = self.transformer(emb).mean(dim=1)
        return emb


# supervised classifier module
class BIOTClassifier(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_classes=6, **kwargs):
        super().__init__()

        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.biot(x)
        x = self.classifier(x)
        return x


# unsupervised pre-train module
class UnsupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_channels=18, **kwargs):
        super(UnsupervisedPretrain, self).__init__()
        self.biot = BIOTEncoder(emb_size, heads, depth, n_channels, **kwargs)
        self.prediction = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )

    def forward(self, x, n_channel_offset=0):
        emb = self.biot(x, n_channel_offset, perturb=True)
        emb = self.prediction(emb)
        pred_emb = self.biot(x, n_channel_offset)
        return emb, pred_emb



# supervised pre-train module
class SupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, **kwargs):
        super().__init__()
        if 'args' in kwargs:
            self.args = kwargs['args']
        else:
            self.args = None
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, args = self.args)
        self.classifier_chb_mit = ClassificationHead(emb_size, 1)
        self.classifier_iiic_seizure = ClassificationHead(emb_size, 6)
        self.classifier_tuab = ClassificationHead(emb_size, 1)
        self.classifier_tuev = ClassificationHead(emb_size, 6)

    def forward(self, x, task="chb-mit"):
        x = self.biot(x)
        if task == "chb-mit":
            x = self.classifier_chb_mit(x)
        elif task == "iiic-seizure":
            x = self.classifier_iiic_seizure(x)
        elif task == "tuab":
            x = self.classifier_tuab(x)
        elif task == "tuev":
            x = self.classifier_tuev(x)
        else:
            raise NotImplementedError
        return x


if __name__ == "__main__":
    x = torch.randn(16, 2, 2000)
    model = BIOTClassifier(n_fft=200, hop_length=200, depth=4, heads=8)
    out = model(x)
    print(out.shape)

    model = UnsupervisedPretrain(n_fft=200, hop_length=200, depth=4, heads=8)
    out1, out2 = model(x)
    print(out1.shape, out2.shape)
