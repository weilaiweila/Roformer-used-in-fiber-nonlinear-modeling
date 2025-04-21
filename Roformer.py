from numpy import dtype
import torch
import torch.nn as nn
import math
from torch import nn, einsum, broadcast_tensors, Tensor
from torch.amp import autocast
from torch.nn import Module, ModuleList
from einops import rearrange, repeat
from numpy import pi
import numpy as np
from typing import Literal
import torch.fft as fft

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast('cuda', enabled = False)
def apply_rotary_emb(
    freqs,
    t,
    dim,
    start_index = 0,
    scale = 1.,
    seq_dim = -2,
    freqs_seq_dim = None
):
    dtype = t.dtype

    if not exists(freqs_seq_dim):
        if freqs.ndim == 2 or t.ndim == 3:
            freqs_seq_dim = 0

    if t.ndim == 3 or exists(freqs_seq_dim):
        seq_len = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim = freqs_seq_dim)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place    
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
        
    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    # theta = torch.arange(0, dim)[:(dim)].float() / dim
    # theta = theta.to(t.device)
    # attenuation = torch.exp(-alpha * theta)
    # out *= attenuation

    return out.type(dtype)


class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        custom_freqs: Tensor | None = None,
        freqs_for:  Literal['lang', 'pixel', 'constant', 'nonlinear', 'nonlinear_dmodel'] = 'lang',
        theta = 10000,#***
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        cache_if_possible = True,
        cache_max_seq_len = 8192
    ):
        super().__init__()


        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        self.dim = dim

        # self.alpha = nn.Parameter(torch.tensor(0.2))

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        elif freqs_for == 'nonlinear':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        # elif freqs_for == 'nonlinear_dmodel':
        #     # a = torch.exp(-0.5 * beta2 * (omega ** 2) * L)
        #     a = 1.
        #     freqs =  a / ((theta * gamma * P * L) ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        self.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_freqs_seq_len = 0

        self.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos

        if not use_xpos:
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base

        self.register_buffer('scale', scale, persistent = False)
        self.register_buffer('cached_scales', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_scales_seq_len = 0

        # add apply_rotary_emb as static method

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device
    
    def get_seq_pos(self, seq_len, device, dtype, offset = 0):
        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0, scale = None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or exists(scale), 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset)

        freqs = self.forward(seq, seq_len = seq_len, offset = offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return apply_rotary_emb(freqs, t, self.dim, scale = default(scale, 1.), seq_dim = seq_dim)
    
    @autocast('cuda', enabled = False)
    def forward(
        self,
        t: Tensor,
        seq_len: int | None = None,
        offset = 0
    ):
        should_cache = (
            self.cache_if_possible and
            not self.learned_freq and
            exists(seq_len) and
            self.freqs_for != 'pixel' and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and \
            exists(self.cached_freqs) and \
            (offset + seq_len) <= self.cached_freqs_seq_len
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        if should_cache and offset == 0:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len = seq_len

        return freqs


 
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, q, k, v, mask = None):
        b, head, s, a_head_length = q.size()
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(a_head_length)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        score = self.softmax(score)
        v = score @ v
        return v, score
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.rotray = RotaryEmbedding(d_model // n_head)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask = None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)

        q, k = self.rotray.rotate_queries_or_keys(q), self.rotray.rotate_queries_or_keys(k)

        out, attn_weights = self.attention(q, k, v, mask)

        out = self.concat(out)
        out = self.w_concat(out)

        return out, attn_weights
    def split(self, x):
        b, s, c = x.size()

        a_head_length = c // self.n_head
        x = x.view(b, s, self.n_head, a_head_length).transpose(1, 2)

        return x
    
    def concat(self, x):
        b, head, s, a_head_length = x.size()
        d_model = head * a_head_length

        x = x.transpose(1, 2).reshape(b, s, d_model) 
        return x
    
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout):
        super(PositionwiseFeedForward, self).__init__()
        # self.d_model = first_model
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.coreNet = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.coreNet(x)

class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_head, ffn_dim, drop_prob):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.attention = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionwiseFeedForward(d_model, ffn_dim, drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask = None):
        out1, attn_weights = self.attention(x, x, x, mask)
        out1 = self.dropout1(out1)
        out1 = self.norm1(x + out1)
        
        out2 = self.ffn(out1)
        out2 = self.dropout2(out2)
        out2 = self.norm2(out1 + out2)
        return out2, attn_weights