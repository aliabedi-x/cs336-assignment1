import torch
import math
from einops import einsum, rearrange
from torch import nn



class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std,b=3 * std) # in-place operation

    def forward(self, x:torch.Tensor):
        Y = einsum(x, self.weight, "... din, dout din -> ... dout")

        return Y

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
    def forward(self, token_ids):
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = (x ** 2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        normalized = x/rms
        result = normalized * self.weight.to(torch.float32)
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = max(64, round((8 * d_model / 3) / 64) * 64)
        
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)


    def forward(self, x:torch.Tensor):
        z = self.w1(x)
        silu = z * torch.sigmoid(z)
        z3 = self.w3(x)
        gate = silu * z3
        result = self.w2(gate)

        return result

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 ==0
        k = torch.arange(d_k//2, dtype=torch.float32, device=device)
        omega_k = theta ** (-2 * k / d_k)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = positions[:, None] * omega_k[None, :]
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(self, x:torch.Tensor, token_positions: torch.Tensor):
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_rot =(x_even*cos) - (x_odd*sin)
        x_odd_rot = (x_even*sin) + (x_odd*cos)
        result = torch.stack((x_even_rot,x_odd_rot), dim=-1)
        return result.flatten(start_dim=-2)

def softmax(x:torch.Tensor, dim:int):
    maximum = torch.amax(x, dim=dim, keepdim=True)
    shifted = x - maximum
    exp = torch.exp(shifted)
    result = exp /(torch.sum(exp, dim=dim, keepdim=True))
    return result

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2,-1)
    scaled_scores = scores/math.sqrt(d_k)
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(~mask, float("-inf"))
    attention_probs = softmax(scaled_scores, dim=-1)
    output = attention_probs@V

    return output

class multihead_self_attention(nn.Module):
    def __init__(self,d_model,num_heads,max_seq_len=None,theta=None,device=None,dtype=None):
        super().__init__()

        assert d_model%num_heads == 0
        if max_seq_len is not None and theta is not None:
            assert (d_model//num_heads)%2 == 0
        self.Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.O = Linear(d_model, d_model, device=device, dtype=dtype)
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = None
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:

        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        q = rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(k, "... s (h d) -> ... h s d", h=self.num_heads)
        v = rearrange(v, "... s (h d) -> ... h s d", h=self.num_heads)

        seq_len = x.shape[-2]
        if token_positions is None: 
            token_positions = torch.arange(seq_len, device=x.device)
        token_positions = token_positions.unsqueeze(-2)
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        head_outputs = scaled_dot_product_attention(q, k, v, mask)
        combined = rearrange(head_outputs,"... h s d -> ... s (h d)")
        o = self.O(combined)

        return o

