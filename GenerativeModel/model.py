import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, fps):
        x_pos_encoded = torch.zeros(x.shape).to(x.device)
        for i in range(len(fps)):
            step_size = round(300/fps[i].item())
            x_pos_encoded[i] = x[i] + self.pe[:x.size(1)*step_size:step_size]
        return x_pos_encoded

class TransformerModel(nn.Module):
    def __init__(self, d_input, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Linear(d_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model, 50000)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(d_model, d_input)
        self.learnable_token = nn.Parameter(torch.zeros(d_input))

    def forward(self, src, mask=None, token_mask=None, fps=None):
        learnable_token = self.learnable_token.unsqueeze(0).unsqueeze(0).repeat(src.shape[0], src.shape[1], 1)
        src = torch.where(token_mask == 0, learnable_token, src)
        src = self.embedding(src)
        src = self.pos_encoder(src, fps=fps)
        output = self.transformer_encoder(src, mask=mask, is_causal=True)
        return self.output(output)
    
    def generate(self, src, max_sequence_length, token_mask, fps, use_mask_on_generation=False):
        device = next(self.parameters()).device
        variance = 0 / 1000
        while src.shape[1] < max_sequence_length:
            src = src.detach()
            mask = generate_square_subsequent_mask(src.shape[1]).to(device)
            output = self(src, mask=mask, token_mask=token_mask[:, :src.shape[1]], fps=fps)
            randomness = (torch.randn(output.shape[0], 1, output.shape[2]) * variance).to(device)
            src = torch.concat((src, output[:, -1:] + randomness), 1) 
        output = torch.concat((src[:, :1], output[:, :-1]), 1)
        if use_mask_on_generation:
            output = output * token_mask[:, :output.shape[1]]
        return output

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask