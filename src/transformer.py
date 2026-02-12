from torch import nn
from constants import d_model, num_layers
from layers import EncoderLayer, DecoderLayer, PositionalEncoder, LayerNormalization


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        # 1. Embedding Layers: Chuyển đổi one-hot token indices sang vector liên tục
        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, d_model)
        
        # 2. Positional Encoding: Tiêm thông tin vị trí vào embeddings
        self.positional_encoder = PositionalEncoder()
        
        # 3. Core Architecture
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        # 4. Chiếu hidden vector về lại không gian từ vựng đích
        self.output_linear = nn.Linear(d_model, self.trg_vocab_size)
        
        # 5. Softmax để chuyển đổi logits sang xác suất
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_input, trg_input, e_mask=None, d_mask=None):
        src_input = self.src_embedding(src_input) 
        trg_input = self.trg_embedding(trg_input) 
        src_input = self.positional_encoder(src_input) 
        trg_input = self.positional_encoder(trg_input) 

        e_output = self.encoder(src_input, e_mask) 
        d_output = self.decoder(trg_input, e_output, e_mask, d_mask) 

        output = self.softmax(self.output_linear(d_output)) 

        return output


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for i in range(num_layers)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, e_mask):
        # Truyền dữ liệu qua từng lớp Encoder tuần tự
        for i in range(num_layers):
            x = self.layers[i](x, e_mask)

        # Chuẩn hóa đầu ra cuối cùng trước khi gửi sang Decoder
        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for i in range(num_layers)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, e_output, e_mask, d_mask):
        # Truyền dữ liệu qua từng lớp Decoder
        # Lưu ý: Decoder cần cả e_output (từ Encoder) và các masks
        for i in range(num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask)

        # Chuẩn hóa đầu ra cuối cùng
        return self.layer_norm(x)
