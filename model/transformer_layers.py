import math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1, atten_dim ='init'):
        super().__init__()
        self.d_model = hidden_dim
        self.atten_dim = atten_dim  #['batch', 'patch','channel','both]
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        if self.atten_dim =='init':
            src = src.view(B*N, L, D)
            # src = src.transpose(0, 1)  #思想是在L维度上做了注意力，就是时序上，因为batch_first = False
            output = self.transformer_encoder(src, mask=None)
            # output = output.transpose(0, 1).view(B, N, L, D) #适用于batch_first=False
            output = output.view(B, N, L, D)
        elif self.atten_dim =='patch':
            src = src.view(B*N, L, D)
            output = self.transformer_encoder(src, mask=None)
            output = output.view(B, N, L, D)
        elif self.atten_dim =='both':
            src = src.view(B,N* L, D)
            output = self.transformer_encoder(src, mask=None)
            output = output.view(B, N, L, D)
        else:
            raise f'not implement {self.atten_dim} check input mode'
        return output
