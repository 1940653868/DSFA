from os import error
import torch
import torch.nn as nn
import numpy as np

# from pre_utils import generate_mask, master_print, master_save, get_loss, _load_data
try:
    from .mask import MaskGenerator
    from .positional_encoding import PositionalEncoding
    from .transformer_layers import TransformerLayers
except:
    from mask import MaskGenerator
    from positional_encoding import PositionalEncoding
    from transformer_layers import TransformerLayers
    
def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class ChanSeqPosiionEmbedding(nn.Module):
    def __init__(self, ch_num, d_model):
        super(ChanSeqPosiionEmbedding, self).__init__()
        self.channel_pos_encoding = nn.Parameter(torch.randn(ch_num, 1, 1), requires_grad=True)
        self.seq_positional_encoding = SeqPositionalEncoding(d_model)
    def forward(self, x):
        x += self.channel_pos_encoding.repeat(1, 1, x.shape[-1])
        x = self.seq_positional_encoding(x)
        return x

class InputEmbedding(nn.Module):
    def __init__(self, in_dim, c_dim, patch_num, d_model, band_num, project_mode, learnable_mask, args, position= None):
        super(InputEmbedding, self).__init__()
        # [batch, channel, ts,emb=256]
        self.mode = project_mode
        self.band_num = band_num
        self.mask_ratio = args.mask_ratio
        self.band_encoding = nn.Parameter(torch.randn(band_num, d_model), requires_grad=True)       # learnable band encoding
        self.positional_encoding = nn.Parameter(torch.randn(patch_num, d_model), requires_grad=True)  # learnable positional encoding
        # self.channel_pos_encoding = nn.Parameter(torch.randn(args.ch_num, 1, 1), requires_grad=True)
        # self.seq_positional_encoding = SeqPositionalEncoding(d_model)
        # self.chanseq_pos = ChanSeqPosiionEmbedding(args.ch_num, d_model)
        if position != None:
            self.chanseq_pos = position
        else:    
            self.chanseq_pos = ChanSeqPosiionEmbedding(args.ch_num,d_model)
        
        if learnable_mask:
            self.mask_encoding = nn.Parameter(torch.randn(in_dim), requires_grad=True)
            self.power_mask_encoding = nn.Parameter(torch.randn(d_model), requires_grad=True)
        else:
            self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
            self.power_mask_encoding = nn.Parameter(torch.zeros(d_model), requires_grad=False)
        
        self.mask_mode =  args.mask_mode  #"patch" #"channel"
        if self.mask_mode == 'patch':
            self.mask = MaskGenerator(num_tokens = patch_num, mask_ratio = self.mask_ratio)
        else:
            self.mask = MaskGenerator(num_tokens = args.ch_num, mask_ratio = self.mask_ratio)
        
        self.softmax = nn.Softmax(dim=-1)
        if project_mode == 'cnn':
            self.cnn = nn.Sequential(
                nn.Conv1d(1, c_dim, 150, 10),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.MaxPool1d(4, 2),
                nn.Conv1d(c_dim, c_dim*2, 10, 5),
                nn.ReLU(inplace=True),  
                nn.Dropout(0.5),
                nn.MaxPool1d(4, 2),
            )

            self.cnn_proj = nn.Sequential(
                nn.Linear(c_dim*2, d_model),
            )
        elif project_mode == 'linear':
            self.proj = nn.Sequential(
                nn.Linear(in_dim, d_model),
            )
        else:
            raise NotImplementedError

        self.apply(_weights_init)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_mask = ids_shuffle[:,len_keep:]
        ids_keep = ids_shuffle[:, :len_keep]
        
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index = ids_restore)

        return x_masked, mask, ids_mask, ids_restore

    def forward(self, data, power, need_mask, mask_by_ch, rand_mask, use_power, use_position = True):
        bat_size, ch_num, patch_num, seg_len = data.shape
        
        if use_power:
            power = self.softmax(power)
            power_emb = torch.einsum('hijk, kl->hijl', power, self.band_encoding)
        # projection
        if self.mode == 'cnn':
            data = data.view(bat_size*ch_num*patch_num, 1, seg_len)
            input_emb = torch.mean(self.cnn(data), dim=-1)
            input_emb = self.cnn_proj(input_emb)
            input_emb = torch.transpose(input_emb, 1, 2)
        elif self.mode == 'linear':
            input_emb = self.proj(data)
            # shape [bat_size*ch_num*patch_num, 1, hidden_dim]
        input_emb = input_emb.view(bat_size, ch_num, patch_num, -1)
        # shape [bat_size,ch_num,patch_num, dimension]
        
        
        # add position embedding
        if use_position:
            # print(input_emb.shape, self.channel_pos_encoding.shape)
            # v1
            # input_emb += self.channel_pos_encoding.repeat(1, 1, input_emb.shape[-1])
            # input_emb = self.seq_positional_encoding(input_emb)
            # v2
            input_emb = self.chanseq_pos(input_emb)
        input_emb = input_emb.view(bat_size, ch_num, patch_num, -1)
        
        bat_size, ch_num, patch_num, fea_dim = input_emb.shape
        #----------------------mask---------------------------------------
        if need_mask:
            x = input_emb.view(bat_size*ch_num, patch_num, fea_dim)
            x_mask, mask, ids_mask, ids_restore = self.random_masking(x, mask_ratio=self.mask_ratio)
            x_mask = x_mask.view(bat_size, ch_num, -1, fea_dim)
            return x_mask, mask, ids_mask, ids_restore
        else:
            x = input_emb.view(bat_size*ch_num, patch_num, fea_dim)
            x_mask, mask, ids_mask, ids_restore = self.random_masking(x, mask_ratio=0)
            x_mask = x_mask.view(bat_size, ch_num, -1, fea_dim)
            # print(x_mask, mask, ids_mask, ids_restore)
            return x_mask, mask, ids_mask, ids_restore
        # #------------old-mask method--------
        # mask_pow = False    
        # if need_mask:
        #     masked_x = input_emb.clone()
        #     if self.mask_mode == 'patch':
        #             unmasked_token_index, masked_token_index = self.mask()
        #             masked_x = masked_x[:,:,unmasked_token_index,: ]

        #             if use_power and mask_pow:
        #                 power_emb = power_emb[:,:,unmasked_token_index,:]
        #     elif self.mask_mode =='channel':
        #             unmasked_token_index, masked_token_index = self.mask()
        #             masked_x = masked_x[:,unmasked_token_index,:,: ]

        #             if use_power and mask_pow:
        #                 # power_emb = power_emb.reshape(bat_size, ch_num * patch_num, -1)
        #                 power_emb = power_emb[:,unmasked_token_index,:,: ]
        # else:
        #     unmasked_token_index, masked_token_index = None, None
        #     masked_x = input_emb
            
        # return masked_x, unmasked_token_index, masked_token_index


class TimeEncoder(nn.Module):
    def __init__(self, in_dim, d_model, dim_feedforward, patch_num, n_layer, nhead, band_num, project_mode, learnable_mask, args=None, postion=None):
        super(TimeEncoder, self).__init__()

        # input_embedding完成了映射 + 时序编码
        self.input_embedding = InputEmbedding(in_dim=in_dim, c_dim=args.ch_num, patch_num = patch_num, d_model=d_model, 
                                band_num=band_num, project_mode=project_mode, learnable_mask=learnable_mask, 
                                args = args, position = postion)
        # N层transformer
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.trans_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layer)

        self.apply(_weights_init)

    def forward(self, data, power, need_mask=True, mask_by_ch=False, rand_mask=True,  use_power=False):
        
        masked_x, mask, ids_mask, ids_restore = self.input_embedding(data, power, need_mask, mask_by_ch, rand_mask, use_power)
        
        batch, channel, patch, d_model =  masked_x.shape
        masked_x = masked_x.view(batch*channel,  patch, d_model)
        trans_out = self.trans_enc(masked_x)
        trans_out = trans_out.view(batch, channel,  patch, d_model)
        return trans_out, mask, ids_mask, ids_restore


class ChannelEncoder(nn.Module):
    def __init__(self, out_dim, d_model, dim_feedforward, n_layer, nhead):
        super(ChannelEncoder, self).__init__()

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.trans_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.apply(_weights_init)

    def forward(self, time_z):
        # time_z.shape: bat_size*seq_len, ch_num, d_model
        batch_size, ch_num, patch_num, d_model = time_z.shape    
        time_z = torch.transpose(time_z, 1, 2)                      
        time_z = time_z.reshape(batch_size*patch_num, ch_num, d_model)         
        ch_out = self.trans_enc(time_z)
        ch_out = ch_out.reshape(batch_size, patch_num, ch_num, -1).transpose(1, 2)
        return ch_out

import math
class SeqPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(SeqPositionalEncoding, self).__init__()
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
            x: `embeddings`, shape (batch,num_chanel, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(-2)]
        return self.dropout(x)



def reshape_tensor_with_overlap(input_tensor, seg_len, stride):
    # 获取原始张量的维度信息
    # batch_size, num_channels,
    seq_len = input_tensor.size()[2]
    # print(input_tensor.shape, seq_len, seg_len, stride) 
    # 计算patch的个数
    patch_num = (seq_len - seg_len)//stride  
    assert(seq_len - seg_len) % stride==0,  f'error'
    # 重新调整张量的形状
    reshaped_tensor = input_tensor.unfold(2, seg_len, stride)

    return reshaped_tensor

class TimeseqUnfoldProject(nn.Module):
    def __init__(self, seg_len, stride, emb_size):
        super(TimeseqUnfoldProject, self).__init__()
        self.seg_len = seg_len
        self.stride = stride
        self.emb_size = emb_size
        self.ProjectLayer = nn.Linear(seg_len, emb_size)

    def forward(self, x):
        x_unfold = reshape_tensor_with_overlap(input_tensor=x, seg_len=self.seg_len,
                    stride=self.stride)
        output = self.ProjectLayer(x_unfold)
        return output, x_unfold
        
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


class Brant_encoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.d_model = args.d_model
        self.seg_len = args.seg_len
        self.stride = args.stride
        # FFT part
        emb_size = 256
        self.emb_size = emb_size
        args.emb_size = emb_size
        self.n_fft = args.seg_len
        self.hop_length = args.seg_len
        args.patch_num = int(1+(args.seq_len - self.n_fft)//self.hop_length)
        # sft-project + time-project --> Tencod(project-transformer)-->
        position = ChanSeqPosiionEmbedding(ch_num= args.ch_num, d_model = self.d_model)
        self.time_enc = TimeEncoder(in_dim          =args.emb_size,
                                    d_model         =args.d_model,
                                    dim_feedforward =args.dim_feedforward,
                                    patch_num       =args.patch_num,  # 需要维护修正，暂时比较混乱
                                    n_layer         =args.time_ar_layer,
                                    nhead           =args.time_ar_head,
                                    band_num        =args.band_num,
                                    project_mode    =args.input_emb_mode,
                                    learnable_mask  =args.learnable_mask,
                                    args = args,
                                    postion=position)
        
        self.ch_enc = ChannelEncoder(out_dim        =args.seg_len,
                                    d_model         =args.d_model,
                                    dim_feedforward =args.dim_feedforward,
                                    n_layer         =args.ch_ar_layer,
                                    nhead           =args.ch_ar_head)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, args.d_model))
        self.positional_encoding = PositionalEncoding(args.d_model, dropout=0.1)
        self.train_mode = args.train_mode   #决定是否掩码，模型输出中间表征还是时序
        self.encdata_mode = args.encdata_mode #'time_only'     #[both, time_only, freq_only] 决定输入给模型进行重构的信号
        print('input content:',  self.encdata_mode)
        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )
        # encode Timeseq to alignment the shape of freq-data
        self.timesequnfold = TimeseqUnfoldProject(
            seg_len=self.n_fft, stride=self.hop_length, emb_size=emb_size)
        # channel token, N_channels >= your actual channels
        
        self.channel_tokens = nn.Embedding(args.ch_num, emb_size)
        self.index = nn.Parameter(
            torch.LongTensor(range(args.ch_num)), requires_grad=False
        )
        # self.seq_positional_encoding = SeqPositionalEncoding(emb_size)

        # norm layers
        self.encoder_norm = nn.LayerNorm(self.d_model)
        
        self.enc_to_dec = nn.Linear(self.d_model, self.d_model)
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, args.ch_num ,args.patch_num, self.d_model), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed = SeqPositionalEncoding(self.d_model) #略于上一个方案，收敛域0.999?
        self.decoder_pos_embed = position
        self.decoder = TransformerLayers(hidden_dim= args.d_model, nlayers=args.decoder_layer, mlp_ratio=1, num_heads=8, dropout=0.1) #dropout 1 waht the fuck
 
        self.decoder_norm = nn.LayerNorm(self.d_model)
        
        if self.encdata_mode == 'freq_only': 
            out_dim = emb_size # 由于单独只有频率的时候，loss的两个输入，重构为[batch，c, path, emb_size] vs [batch, c, patch, out_dim]
        else:
            out_dim = args.seg_len
        self.output_layer = nn.Linear(self.d_model, out_dim)

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, masked_token_index):
        """Get reconstructed masked tokens and corresponding ground-truth for subsequent loss computing.

        Args:
            reconstruction_full (torch.Tensor): reconstructed full tokens.
            real_value_full (torch.Tensor): ground truth full tokens.
            unmasked_token_index (list): unmasked token index.
            masked_token_index (list): masked token index.

        Returns:
            torch.Tensor: reconstructed masked tokens.
            torch.Tensor: ground truth masked tokens.
        """
        # get reconstructed masked tokens
        batch_size, num_nodes, _, _ = reconstruction_full.shape
        # reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]     # B, N, r*P, d
        reconstruction_masked_tokens = reconstruction_full[:, :, masked_token_index, :].contiguous()
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)     # B, r*P*d, N
        # reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size*num_nodes, -1)
        
        
        # in lateset version code, the real_value_full shape:[batch, channel, seq_len], but now we unfold data at the beginning 
        # so we don't need to unfold it here again
        # print('full value', real_value_full.shape)
        # label_full = real_value_full.unfold(2, self.seg_len, self.seg_len)  # B, N, P, L
        label_full = real_value_full
        label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous() # B, N, r*P, d
        label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)  # B, r*P*d, N
        
        # label_masked_tokens = label_masked_tokens.view(batch_size* num_nodes, -1)

        return reconstruction_masked_tokens, label_masked_tokens    
    
        # rec_down_samp_rate = in_dim // out_dim
    
    def stft(self, sample):
        spectral = torch.stft( 
            input = sample.squeeze(1),
            n_fft = self.n_fft,             #FFT的大小决定了频率分辨率,较大的n_fft能够获得更好的频率分辨率,但也会增加计算量
            hop_length = self.hop_length,   #这个参数指定了STFT中相邻窗口之间的步长(以样本为单位)。较小的hop_length会产生更多的重叠,从而获得更好的时间分辨率
            window = torch.hamming_window(window_length=self.hop_length).to('cuda'), # 新添加的
            center = False,
            onesided = True,
            return_complex = True,
            normalized=False
        )
        # num_frames 取决于信号长度和 hop_length。num_fft_bins 通常为 n_fft // 2 + 1,因为只返回单边频谱。
        # T is the number of frames, 1 + L // hop_length for center=True, or 1 + (L - n_fft) // hop_length otherwise.
        return torch.abs(spectral), torch.angle(spectral) #复数返回张量 #[batch,n_fft//2+1, block_num=(length-hop)//hop]

    def decoding(self,  hidden_states_unmasked, masked_token_index):
        """Decoding process : encoder 2 decoder layer, add mask tokens, Transformer layers, predict.

        Args:
            hidden_states_unmasked (torch.Tensor): hidden states of masked tokens [B, N, P*(1-r), d].
            masked_token_index (list): masked token index

        Returns:
            torch.Tensor: reconstructed data
        """
        batch_size, num_nodes, _, _ = hidden_states_unmasked.shape
        # encoder 2 decoder layer
        
        # add mask tokens,在维度上实现对其，期望decoder发现问题，重构最后mask-token内的数值
        hidden_states_masked = self.positional_encoding(
            self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), hidden_states_unmasked.shape[-1]),
            index = masked_token_index
            )
        # 这个步骤不正确?，缺乏了对齐位置编码的设置
        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)   # B, N, P, d
        
        # decoding
        hidden_states_full = self.decoder(hidden_states_full)
        hidden_states_full = self.decoder_norm(hidden_states_full)

        # prediction (reconstruction)
        reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.d_model))

        return reconstruction_full
        

    def forward_decoder(self, x, ids_restore):
        batch_size, ch_num, pa_num, dimen = x.shape
        x = x.view(batch_size*ch_num, pa_num, dimen)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x  = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = x.reshape(batch_size, ch_num, -1, dimen)
        # x = x + self.decoder_pos_embed
        x = self.decoder_pos_embed(x)

        # decoding
        hidden_states_full = self.decoder(x)
        hidden_states_full = self.decoder_norm(hidden_states_full)

        # prediction (reconstruction)
        reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, ch_num, -1, dimen))

        return reconstruction_full

    def forward(self, batch, use_power,
                mask_by_ch, rand_mask):
        
        time_seq = batch[0]
        power = batch[1]
        
        if self.encdata_mode =='time_only':
            use_sft = False
        else:
            use_sft = True
        #data_mode:time_only, freq_only, time_freq
        if use_sft: #计算power用于下游
            emb_power_seq = []
            init_power = []
            n_channel_offset = 0
            x = power
            batch_size, channel_num, _ = x.shape
            for i in range(x.shape[1]): # x.shape ([batch, channel, time_seq])
                channel_spec_emb, channel_angle_emb = self.stft(x[:, i : i + 1, :])            # [batch, fre_num, time_blcok_num]
                channel_spec_emb = self.patch_embedding(channel_spec_emb)   # Size([batch, time_blcok_num, fea_dim])
                batch_size, ts, _ = channel_spec_emb.shape
                # (batch_size, ts, emb)
                channel_token_emb = (
                    self.channel_tokens(self.index[i + n_channel_offset])
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(batch_size, ts, 1)
                )   # (batch_size, ts, emb)
                init_power.append(channel_spec_emb)            
                # channel_emb = self.seq_positional_encoding(channel_spec_emb + channel_token_emb)
                channel_emb = channel_spec_emb # + channel_token_emb #暂时跳过，避免重复embedding
                # perturb
                perturb = False
                if perturb:
                    ts = channel_emb.shape[1]
                    ts_new = np.random.randint(ts // 2, ts)
                    selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                    channel_emb = channel_emb[:, selected_ts]
                emb_power_seq.append(channel_emb)
            emb = torch.cat(emb_power_seq, dim=1)  #[batch_size, channel*ts_blcok,emb_size ]
            emb  = emb.reshape(batch_size, channel_num, -1, self.emb_size)
            init_fre_emb = torch.cat(init_power, dim=1)
            init_fre_emb = init_fre_emb.reshape(batch_size, channel_num, -1, self.emb_size)
        # Align the time-seq to fft-seq 
        time_data, time_fold = self.timesequnfold(time_seq)
        # decide the input data content
        if self.encdata_mode == 'time_only':
            data = time_data
            # data pass to model
            initdata = time_fold.detach().clone()
            # initdata pass to reconstruct
        elif self.encdata_mode == 'freq_only':
            data = init_fre_emb 
            initdata = init_fre_emb.detach().clone()
        elif self.encdata_mode == 'both':
            data = init_fre_emb + time_data
            initdata = (init_fre_emb + time_data).detach().clone()
        batch_size, ch_num, patch_num, seg_len = data.shape

        need_mask = True if self.train_mode == 'pretrain' else False
        # torch.Size([352, 7, 1280]) unmask-7 mask-2
        time_z, mask, ids_mask, ids_restore = self.time_enc(data, power,
                                                            need_mask=need_mask,
                                                            mask_by_ch=mask_by_ch,
                                                            rand_mask=rand_mask,
                                                            use_power=use_power)  

        batch_size, ch_num, patch_num, d_model = time_z.shape    
        # channel encoder
        ch_out = self.ch_enc(time_z)             # rec.shape: batch_size*patch_num, ch_num, seg_len //  #torch.Size([16, 22, 7, d_dim])   
        ch_out = self.encoder_norm(ch_out)
        if self.train_mode == 'pretrain':
            ch_out = self.enc_to_dec(ch_out)

            # reconstruction_full = self.decoding(ch_out, masked_token_index=masked_token_index) #torch.Size([16, 22, 9, 125])
            reconstruction_full = self.forward_decoder(ch_out, ids_restore)
        
        
            # for subsequent loss computing  
            # bug: initdata 在第一次时域频域对其的操作上已经被修改，需要找回原来的输入
            # initdata = reshape_tensor_with_overlap(input_tensor=batch[0], seg_len=self.seg_len, stride=self.stride)
            # 现在频域，initdata是经过了MLP映射后的；时域数据如果经过reshape，确保最后一层125维度，则是原始的
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, initdata,  masked_token_index = ids_mask)
            
            return reconstruction_masked_tokens, label_masked_tokens
            
        else:
            # rec = rec.view(batch_size, patch_num, ch_num, seg_len // )
            # rec = torch.transpose(rec, 1, 2)    # transpose back  rec.shape: batch_size, ch_num, patch_num, seg_len
            # rec = rec.reshape(batch_size*ch_num*patch_num, seg_len // )

            # data = data.view(batch_size * ch_num * patch_num, seg_len)[::]
            # return data, rec 
            ch_out = torch.mean(ch_out, dim=-2)
            return ch_out







    
if __name__ =="__main__":
    import argparse   
    from torchinfo import summary
    import os
    num_gpus = torch.cuda.device_count()
    # 设置CUDA_VISIBLE_DEVICES环境变量为最后一个GPU的索引
    os.environ["CUDA_VISIBLE_DEVICES"] = str(num_gpus - 1)

    # 确保只有一个GPU设备可见
    print("Visible GPUs:", os.environ["CUDA_VISIBLE_DEVICES"])

    # 现在你可以像平常那样初始化你的深度学习库（如PyTorch或TensorFlow），它们将只会看到你指定的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    parser = argparse.ArgumentParser()
    
    def set_args_func():
        import argparse   
        from torchinfo import summary
    
        
        def str2bool(v):
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Unsupported value encountered.')
        parser = argparse.ArgumentParser()

        if  torch.cuda.is_available():
            # Set available CUDA devices
            # This option is crucial for multiple GPUs
            # 'cuda' ≡ 'cuda:0'
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        parser.add_argument("--device", type=torch.device, default=device)
        parser.add_argument("--gpu_id", type=int, default=0)

        parser.add_argument('--local_rank', type=int, default=-1)
        parser.add_argument('--dev_ids', type=str, default="0,1,2,3,4,5,6,7")
        parser.add_argument("--use_power", type=bool, default=False)
        
        parser.add_argument("--band_num", type=int, default=8)
        parser.add_argument("--d_model", type=int, default=320)
        seq_len = 1125
        seg_len = 45
        parser.add_argument("--encdata_mode", type=str, default='freq_only', choices=['time_only','freq_only','both'])
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--seq_len", type=int, default = seq_len)
        parser.add_argument("--seg_len", type=int, default = seg_len)
        parser.add_argument("--stride", type=int, default = seg_len)
        parser.add_argument("--ch_num", type=int, default=22)
        patch_num = int((seq_len-seg_len)/seg_len+1)
        parser.add_argument("--patch_num", type=int, default=patch_num)
        parser.add_argument("--time_ar_layer", type=int, default=4)#8
        parser.add_argument("--time_ar_head", type=int, default=16)
        parser.add_argument("--input_emb_mode", type=str, default='linear')
        parser.add_argument("--ch_ar_layer", type=int, default=4)#5
        parser.add_argument("--ch_ar_head", type=int, default=16)
        parser.add_argument("--decoder_layer", type=int, default=4)#5
        parser.add_argument("--learnable_mask", type=str2bool, default=False)
        parser.add_argument('--train_mode', type=str, default = 'pretrain') #finetune
        parser.add_argument("--dist_data_parallel", type=str2bool, default=False)
        parser.add_argument("--amp", type=str2bool, default=False,
                            help='whether to use automatic mixed precision')
        parser.add_argument("--start_epo_idx", type=int, default=-1,
                            help='load the model from last train, -1 means a new model')

        parser.add_argument("--num_epochs", type=int, default=100)
        parser.add_argument("--num_workers", type=int, default=4)

        parser.add_argument("--mask_ratio", type=float, default=0.5)

        parser.add_argument("--optimizer", type=str, default='adam')
        parser.add_argument("--scheduler", type=str, default='cyclic')
        parser.add_argument("--accu_step", type=int, default=4)
        parser.add_argument("--time_lr", type=float, default=3e-6)
        parser.add_argument("--ch_lr",   type=float, default=3e-6)

        parser.add_argument("--mask_mode", type=str, default='patch')
        parser.add_argument("--mask_by_channel", type=bool, default=False)
        parser.add_argument("--rand_mask", type=bool, default=False)
        parser.add_argument("--mask_len", type=int, default=100)
        
        # 超参数部分
        parser.add_argument("--batch_size", type=int, default=16)
        
        args = parser.parse_args()
        return args


    
    args = set_args_func()
    model = Brant_encoder(args).cuda()
    ch_num = args.ch_num
    batch = 7
    mask_rand =  None #generate_mask(ch_num, args.patch_num, args.mask_ratio)
    input_size = (2, batch, ch_num, args.seq_len)
    batch_data = torch.ones(2, batch, ch_num, args.seq_len)
    params = {
        # "batch":batch_data,
        "use_power":args.use_power,
        "mask_by_ch":args.mask_by_channel, 
        "rand_mask":args.rand_mask, 
    }
    print(type(mask_rand))
    summary(model, input_size, **params)    
    # summary(model, **params)    