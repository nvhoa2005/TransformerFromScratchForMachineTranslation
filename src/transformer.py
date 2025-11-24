from torch import nn
from constants import d_model, num_layers
from layers import EncoderLayer, DecoderLayer, PositionalEncoder, LayerNormalization

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for i in range(num_layers)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, e_mask):
        for i in range(num_layers):
            x = self.layers[i](x, e_mask)

        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for i in range(num_layers)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask)

        return self.layer_norm(x)