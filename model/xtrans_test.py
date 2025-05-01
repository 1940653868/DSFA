import os

# 获取可用GPU数量
import torch

import torch
from x_transformers import TransformerWrapper, Decoder, Encoder
num_gpus = torch.cuda.device_count()

# 设置CUDA_VISIBLE_DEVICES环境变量为最后一个GPU的索引
os.environ["CUDA_VISIBLE_DEVICES"] = str(num_gpus - 1)

# 确保只有一个GPU设备可见
print("Visible GPUs:", os.environ["CUDA_VISIBLE_DEVICES"])

# 现在你可以像平常那样初始化你的深度学习库（如PyTorch或TensorFlow），它们将只会看到你指定的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from x_transformers import Decoder, Encoder

enc = Encoder(
    dim = 512,
    depth = 6,
    heads = 8,
    attn_num_mem_kv = 16, # 16 memory key / values,
    attn_flash = True
).to(device)


model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    emb_dropout = 0.1,         # dropout after embedding
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        layer_dropout = 0.1,   # stochastic depth - dropout entire layer
        attn_dropout = 0.1,    # dropout post-attention
        ff_dropout = 0.1,       # feedforward dropout
        attn_flash = True,
    )
).to(device)

x = torch.randint(0, 20000, (1, 1024)).cuda()
y = model(x)
# x = torch.randint(0, 20000, (1, 512)).cuda()
# y = enc(x)
print(y.shape)

import torch
from x_transformers import XTransformer

model = XTransformer(
    dim = 512,
    enc_num_tokens = 256,
    enc_depth = 6,
    enc_heads = 8,
    enc_max_seq_len = 1024,
    dec_num_tokens = 256,
    dec_depth = 6,
    dec_heads = 8,
    dec_max_seq_len = 1024,
    tie_token_emb = True      # tie embeddings of encoder and decoder
)

src = torch.randint(0, 256, (1, 1024))
src_mask = torch.ones_like(src).bool()
tgt = torch.randint(0, 256, (1, 1024))

loss = model(src, tgt, mask = src_mask) # (1, 1024, 512)
loss.backward()
print(loss.item())