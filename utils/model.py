import torch
import torch.nn as nn

import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MultiheadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.5, fp16=False):
        super(MultiheadAttentionLayer, self).__init__()
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        self.scale = np.sqrt(self.head_dim)

        self.mask_fill = -65504 if fp16 else -1e10

    def forward(self, query, key, value, mask=None):
        bs = query.shape[0]
        Q = self.fc_q(query).view(bs, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.fc_k(key).view(bs, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.fc_v(value).view(bs, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None: energy = energy.masked_fill(mask==0, self.mask_fill)

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bs, -1, self.hidden_dim)
        x = self.fc_o(x)
        return x, attention

class PositionFeedForward(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout=0.5):
        super(PositionFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.dropout(torch.relu(self.fc1(x)))
        out = self.fc2(out)
        return out
    
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout=0.5, fp16=False):
        super(EncoderLayer, self).__init__()
        self.at_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiheadAttentionLayer(hidden_dim, n_heads, dropout=dropout, fp16=fp16)
        self.positionwise_ff = PositionFeedForward(hidden_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        _out, _ = self.attention(x, x, x, mask)
        out = self.at_layer_norm(x + self.dropout(_out))
        _out = self.positionwise_ff(out)
        out = self.ff_layer_norm(out + self.dropout(_out))
        return out
    
class Encoder(nn.Module):
    def __init__(self, vocab_sz, hidden_dim, n_layers, n_heads, pf_dim, dropout=0.5, msl=100, fp16=False):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_sz, hidden_dim)
        self.posit_embedding = nn.Embedding(msl, hidden_dim)
        layers = [EncoderLayer(hidden_dim, n_heads, pf_dim, dropout, fp16=fp16) for _ in range(n_layers)]
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(hidden_dim)

    def forward(self, x, mask):
        bs, msl = x.shape
        pos = torch.arange(0, msl).unsqueeze(0).repeat(bs, 1).to(next(self.parameters()).device)
        out = self.token_embedding(x) * self.scale + self.posit_embedding(pos)
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(out, mask)
        return out
    
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout=0.5, fp16=False):
        super(DecoderLayer, self).__init__()
        self.self_at_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.enco_at_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiheadAttentionLayer(hidden_dim, n_heads, dropout, fp16=fp16)
        self.enco_attention = MultiheadAttentionLayer(hidden_dim, n_heads, dropout, fp16=fp16)
        self.positionwise_ff = PositionFeedForward(hidden_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, enc_src, src_mask, trg_mask):
        _out, _ = self.self_attention(y, y, y, trg_mask)
        out = self.self_at_layer_norm(y + self.dropout(_out))
        _out, attention = self.enco_attention(out, enc_src, enc_src, src_mask)
        out = self.enco_at_layer_norm(out + self.dropout(_out))
        _out = self.positionwise_ff(out)
        out = self.ff_layer_norm(out + self.dropout(_out))
        return out, attention
    
class Decoder(nn.Module):
    def __init__(self, vocab_sz, hidden_dim, n_layers, n_heads, pf_dim, dropout, msl=100, fp16=False):
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_sz, hidden_dim)
        self.posit_embedding = nn.Embedding(msl, hidden_dim)
        layers = [DecoderLayer(hidden_dim, n_heads, pf_dim, dropout, fp16=fp16) for _ in range(n_layers)]
        self.layers = nn.ModuleList(layers)
        self.fc1 = nn.Linear(hidden_dim, vocab_sz)
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(hidden_dim)

    def forward(self, y, enc_src, src_mask, trg_mask):
        bs, msl = y.shape
        pos = torch.arange(0, msl).unsqueeze(0).repeat(bs, 1).to(next(self.parameters()).device)
        out = self.token_embedding(y) * self.scale + self.posit_embedding(pos)
        out = self.dropout(out)

        for layer in self.layers:
            out, attention = layer(out, enc_src, src_mask, trg_mask)
        out = self.fc1(out)

        return out, attention
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_ix, trg_pad_ix, tie_weights=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_ix = src_pad_ix
        self.trg_pad_ix = trg_pad_ix

        if tie_weights:
            encoder.token_embedding.weight = decoder.token_embedding.weight
            decoder.fc1.weight = decoder.token_embedding.weight

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, x):
        mask = (x != self.src_pad_ix).unsqueeze(1).unsqueeze(2)
        return mask

    def make_trg_mask(self, y):
        bs, msl = y.shape
        mask = (y != self.trg_pad_ix).unsqueeze(1).unsqueeze(2)
        submask = torch.tril(torch.ones(msl, msl)).bool().to(next(self.parameters()).device)
        mask = mask & submask
        return mask

    def forward(self, x, y):
        src_mask = self.make_src_mask(x)
        trg_mask = self.make_trg_mask(y)

        enc_src = self.encoder(x, src_mask)
        out, attention = self.decoder(y, enc_src, src_mask, trg_mask)
        return out, attention
