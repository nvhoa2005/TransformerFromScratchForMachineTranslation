from torch import nn
from constants import d_model, drop_out_rate, num_heads, d_k, d_ff, seq_len, attention_type

import torch
import math


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_1 = LayerNormalization()
        
        # Chọn loại attention theo tham số cấu hình
        if attention_type == 'bahdanau':
            self.multihead_attention = BahdanauMultiheadAttention()
        elif attention_type == 'scaled_dot_product':
            self.multihead_attention = MultiheadAttention()
        elif attention_type == 'luong':
            self.multihead_attention = LuongMultiheadAttention()
        self.drop_out_1 = nn.Dropout(drop_out_rate)

        self.layer_norm_2 = LayerNormalization()
        self.feed_forward = FeedFowardLayer()
        self.drop_out_2 = nn.Dropout(drop_out_rate)

    def forward(self, x, e_mask):
        # 1. Pre-Norm Self-Attention Sublayer
        x_1 = self.layer_norm_1(x)  # Normalize TRƯỚC khi vào Attention
        x = x + self.drop_out_1(
            self.multihead_attention(x_1, x_1, x_1, mask=e_mask)
        )  # Residual Connection
        
        # 2. Pre-Norm Feed-Forward Sublayer
        x_2 = self.layer_norm_2(x)
        x = x + self.drop_out_2(self.feed_forward(x_2))

        return x


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_1 = LayerNormalization()
        
        # Masked Attention
        self.masked_multihead_attention = None
        if attention_type == 'bahdanau':
            self.masked_multihead_attention = BahdanauMultiheadAttention()
        elif attention_type == 'scaled_dot_product':
            self.masked_multihead_attention = MultiheadAttention()
        elif attention_type == 'luong':
            self.masked_multihead_attention = LuongMultiheadAttention()
        self.drop_out_1 = nn.Dropout(drop_out_rate)

        self.layer_norm_2 = LayerNormalization()
        # Cross Attention: Decoder query lấy thông tin từ Encoder Key/Value
        self.multihead_attention = None
        if attention_type == 'bahdanau':
            self.multihead_attention = BahdanauMultiheadAttention()
        elif attention_type == 'scaled_dot_product':
            self.multihead_attention = MultiheadAttention()
        elif attention_type == 'luong':
            self.multihead_attention = LuongMultiheadAttention()
        self.drop_out_2 = nn.Dropout(drop_out_rate)

        self.layer_norm_3 = LayerNormalization()
        self.feed_forward = FeedFowardLayer()
        self.drop_out_3 = nn.Dropout(drop_out_rate)

    def forward(self, x, e_output, e_mask,  d_mask):
        # 1. Masked Self-Attention
        x_1 = self.layer_norm_1(x)
        x = x + self.drop_out_1(
            self.masked_multihead_attention(x_1, x_1, x_1, mask=d_mask)
        )
        
        # 2. Cross-Attention
        # Query (x_2) đến từ Decoder, Key/Value (e_output) đến từ Encoder
        x_2 = self.layer_norm_2(x)
        x = x + self.drop_out_2(
            self.multihead_attention(x_2, e_output, e_output, mask=e_mask)
        )
        
        # 3. Feed Forward
        x_3 = self.layer_norm_3(x)
        x = x + self.drop_out_3(self.feed_forward(x_3))

        return x


class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.inf = 1e9

        # Chiếu Q, K, V vào không gian ngữ nghĩa
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_out_rate)
        self.attn_softmax = nn.Softmax(dim=-1)

        # Output linear
        self.w_0 = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape

        # Linear projection & Split heads
        # Reshape từ (B, L, d_model) -> (B, L, num_heads, d_k)
        q = self.w_q(q).view(input_shape[0], -1, num_heads, d_k) # (B, L, num_heads, d_k)
        k = self.w_k(k).view(input_shape[0], -1, num_heads, d_k) # (B, L, num_heads, d_k)
        v = self.w_v(v).view(input_shape[0], -1, num_heads, d_k) # (B, L, num_heads, d_k)

        # Transpose để đưa num_heads ra ngoài phục vụ tính toán song song
        # Shape mới: (B, num_heads, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 2. Calculate Attention Score
        attn_values = self.self_attention(q, k, v, mask=mask) # (B, num_heads, L, d_k)
        
        # 3. Concatenate heads & Final Linear
        concat_output = attn_values.transpose(1, 2)\
            .contiguous().view(input_shape[0], -1, d_model) # (B, L, d_model)

        return self.w_0(concat_output)

    def self_attention(self, q, k, v, mask=None):
        # Matmul Q và K^T: (B, H, L, d_k) @ (B, H, d_k, L) -> (B, H, L, L)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) # (B, num_heads, L, L)
        
        # Scaling: Chia cho sqrt(d_k) để tránh dot-product quá lớn.
        attn_scores = attn_scores / math.sqrt(d_k)

        # Masking: Gán giá trị -inf cho các vị trí bị che
        if mask is not None:
            mask = mask.unsqueeze(1) # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax & Weighted Sum
        attn_distribs = self.attn_softmax(attn_scores)
        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v) # (B, num_heads, L, d_k)

        return attn_values


class FeedFowardLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)

        return x


class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        # LayerNorm để chuẩn hóa theo chiều d_model
        self.layer = nn.LayerNorm([d_model], elementwise_affine=True, eps=self.eps)

    def forward(self, x):
        x = self.layer(x)

        return x


class PositionalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Tạo ma trận PE một lần duy nhất
        pe_matrix= torch.zeros(seq_len, d_model) # (L, d_model)

        # Tính position encoding values
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))

        pe_matrix = pe_matrix.unsqueeze(0)
        # self.positional_encoding = pe_matrix.to(device=device).requires_grad_(False)
        self.register_buffer('positional_encoding', pe_matrix)

    def forward(self, x):
        x = x * math.sqrt(d_model) # (B, L, d_model)
        x = x + self.positional_encoding # (B, L, d_model)

        return x

class BahdanauMultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.inf = 1e9
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Bahdanau specific: Vector v là tham số học được
        self.v = nn.Parameter(torch.rand(1, num_heads, 1, 1, d_k))
        nn.init.xavier_uniform_(self.v)

        self.dropout = nn.Dropout(drop_out_rate)
        self.attn_softmax = nn.Softmax(dim=-1)

        # Final output linear transformation
        self.w_0 = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape 

        # Linear & Split heads
        q = self.w_q(q).view(input_shape[0], -1, num_heads, d_k) # (B, L, H, d_k)
        k = self.w_k(k).view(input_shape[0], -1, num_heads, d_k) 
        v = self.w_v(v).view(input_shape[0], -1, num_heads, d_k) 

        # Transpose to (B, H, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        energy = torch.tanh(q.unsqueeze(3) + k.unsqueeze(2)) 
        
        attn_scores = torch.sum(self.v * energy, dim=-1) 

        if mask is not None:
            mask = mask.unsqueeze(1) 
            attn_scores = attn_scores.masked_fill(mask == 0, -1 * self.inf)

        attn_distribs = self.attn_softmax(attn_scores)
        attn_distribs = self.dropout(attn_distribs)
        
        attn_values = torch.matmul(attn_distribs, v) 
        
        concat_output = attn_values.transpose(1, 2)\
            .contiguous().view(input_shape[0], -1, d_model)

        return self.w_0(concat_output)

class LuongMultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.inf = 1e9

        # Standard projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_a = nn.Parameter(torch.rand(num_heads, d_k, d_k))
        nn.init.xavier_uniform_(self.w_a)

        self.dropout = nn.Dropout(drop_out_rate)
        self.attn_softmax = nn.Softmax(dim=-1)

        self.w_0 = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape

        # 1. Linear & Split heads
        q = self.w_q(q).view(input_shape[0], -1, num_heads, d_k) # (B, L_q, H, d_k)
        k = self.w_k(k).view(input_shape[0], -1, num_heads, d_k) # (B, L_k, H, d_k)
        v = self.w_v(v).view(input_shape[0], -1, num_heads, d_k) 

        # Transpose => (B, H, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q_weighted = torch.einsum('bhld,hde->bhle', q, self.w_a)

        attn_scores = torch.matmul(q_weighted, k.transpose(-2, -1)) # (B, H, L_q, L_k)
        
        # 3. Masking & Softmax (Standard)
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1 * self.inf)

        attn_distribs = self.attn_softmax(attn_scores)
        attn_distribs = self.dropout(attn_distribs)
        
        attn_values = torch.matmul(attn_distribs, v)

        concat_output = attn_values.transpose(1, 2)\
            .contiguous().view(input_shape[0], -1, d_model)

        return self.w_0(concat_output)
