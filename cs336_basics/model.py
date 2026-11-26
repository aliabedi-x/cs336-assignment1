import torch
import math
from einops import einsum, rearrange
from torch import nn
from collections.abc import Callable, Iterable
from typing import Optional
import numpy as np



class Linear(nn.Module): # in_features, out_features
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std,b=3 * std) # in-place operation

    def forward(self, x:torch.Tensor):
        Y = einsum(x, self.weight, "... din, dout din -> ... dout")

        return Y

class Embedding(nn.Module): # num_embeddings, embedding_dim
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
    def forward(self, token_ids):
        return self.weight[token_ids]

class RMSNorm(nn.Module): # d_model
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

class SwiGLU(nn.Module): # d_model, d_ff=None
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

class RotaryPositionalEmbedding(nn.Module): # theta, d_k, max_seq_len
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

class multihead_self_attention(nn.Module): # d_model,num_heads,max_seq_len=None,theta=None
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

class TransformerBlock(nn.Module): # d_model, num_heads, d_ff, max_seq_len, theta
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rmsnorm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = multihead_self_attention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ffswiglu = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None):
        y = x + self.attn(self.rmsnorm1(x), token_positions) # attn + RMSNorm

        z = y + self.ffswiglu(self.rmsnorm2(y))

        return z

class TransformerLm(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta, device=None, dtype=None):
        super().__init__()
        # Token IDs(batch, seq_len) -> Token Embedding(batch, seq_len, d_model) -> 
        # -> Transformer Blocks(N number) -> Final RMSNorm -> LM Head -> Vocab Logits
        self.num_layers = num_layers
        self.token_emb = Embedding(vocab_size, d_model, device, dtype)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device=device, dtype=dtype)
            self.blocks.append(block)
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids):
        embedded = self.token_emb(token_ids)
        for block in self.blocks:
            embedded = block(embedded)
        f_norm = self.final_norm(embedded)
        f_lin = self.lm(f_norm)
        return f_lin

def cross_entropy(logits, targets):
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    maximum = torch.amax(flat_logits, dim=-1, keepdim=True)
    shifted = flat_logits - maximum
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    row_indices = torch.arange(flat_logits.shape[0], device=logits.device)
    target_logits = shifted[row_indices, flat_targets]
    losses = log_sum_exp - target_logits
    result = torch.mean(losses)
    return result

def perplexity(logits, targets): # Usually for evaluating the model
    loss = cross_entropy(logits, targets)
    return torch.exp(loss)

# A little bit explanation: for Optimizer we inheritas from torch.optim.Optimizer, why? because we want to update Parameters and this subclass will do it for us.
# torch.optim.Optimizer -> gives some features like: Parameter Groups, holding Optimizer State and save State Dictionary

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr<0:
            raise ValueError(f'Invalid learning rate: {lr}')
        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p] # Get p state
                t = state.get('t', 0) # Get iteration numb from the state, or 0
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                state['t'] = t + 1
        return loss
    
def main():
    #==START==# An small test for different Learning rates. 
    lrs = [1e1, 1e2, 1e3]
    for lr in lrs:
        weights = torch.nn.Parameter(5*torch.randn((10,10)))
        opt = SGD([weights], lr=lr)

        for t in range(10):
            opt.zero_grad()
            loss = (weights ** 2). mean()
            print(loss.cpu().item())
            loss.backward()
            opt.step()
        

        print(f'The loss associated to learning rage={lr} is {loss}')
    #==END==# An small test for different Learning rates. 

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01):
        if lr>=0 and betas[0]<1 and betas[0]>=0 and betas[1]<1 and betas[1]>=0 and eps>=0 and weight_decay>=0: 
            defaults = {'lr': lr, 'betas': betas, 'eps':eps, 'weight_decay': weight_decay}
            super().__init__(params, defaults)
        else:
            raise ValueError(f'Invalid hyperparameter')

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group['params']:
                if p.grad is None:
                    continue
            
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                t = state["t"]
                m = state["m"]
                v = state["v"]

                grad = p.grad.detach()
                m = beta1 * m + (1-beta1)*grad
                v = beta2 * v + (1-beta2)*(grad**2)

                state["m"] = m
                state["v"] = v
                lrt = lr*((math.sqrt(1-(beta2**(t+1)))/(1-(beta1**(t+1)))))
                with torch.no_grad():
                    p.mul_(1 - lr * weight_decay)
                    p.sub_(lrt * m / (torch.sqrt(v) + eps))

                state['t'] = t + 1
        return loss

def learing_rate_schedule(t, a_max, a_min, T_w, T_c): # Here a = alpha in formula
    if t< T_w:
        alpha = (t/T_w)*a_max
    elif T_w<= t <= T_c:
        alpha = a_min + 0.5*(1+math.cos(((t-T_w)/(T_c-T_w))*math.pi))*(a_max-a_min)
    else: 
        alpha = a_min
    
    return alpha

def gradient_clipping(parameters, maximum_norm, eps=1e-6):
    parameters = list(parameters)
    total_squared_norm = 0
    for p in parameters:
        if p.grad is None:
            continue
        squared_sum = torch.sum(torch.square(p.grad))
        total_squared_norm += squared_sum
    global_norm = math.sqrt(total_squared_norm)
    if global_norm > maximum_norm:
        c = maximum_norm/(global_norm+eps)
        with torch.no_grad():
            for p in parameters:
                if p.grad is None:
                    continue
                p.grad.mul_(c)

def data_loading(x, batch_size, context_length, device=None):
    N = len(x)
    start_index = np.random.randint(0, N-context_length, batch_size)
    input_sequence = []
    target_sequence  = []
    for index in start_index:
        input_sequence.append(x[ index : index+context_length ])
        target_sequence.append(x[ index+1 : index+context_length+1 ])
    data_input = np.stack(input_sequence, axis=0)
    data_target = np.stack(target_sequence, axis=0)
    result_input = torch.tensor(data_input, dtype=torch.long, device=device)
    result_target = torch.tensor(data_target, dtype=torch.long, device=device)
    return (result_input,result_target)

def save_checkpoint(model, optimizer, iteration, out):
    model_data = model.state_dict()
    opt_data = optimizer.state_dict()
    data = {"model_data": model_data, "opt_data": opt_data, "iteration":iteration}
    torch.save(data,out)

def load_checkpoint(src, model, optimizer):
    data = torch.load(src, map_location=None)
    model.load_state_dict(data["model_data"])
    optimizer.load_state_dict(data["opt_data"])
    return data["iteration"]

def SiLU(x: torch.Tensor) -> torch.Tensor:
    silu = x * torch.sigmoid(x)
    return silu

if __name__ == '__main__':
    main()
