import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


PALIGEMMA_VOCAB_SIZE = 257_152


class Config:
    def __init__(self, width: int, depth: int, mlp_dim: int, num_heads: int, num_kv_heads: int, head_dim: int):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


gemma_2b_config = Config(
    width=2048,
    depth=18,
    mlp_dim=16_384,
    num_heads=8,
    num_kv_heads=1,
    head_dim=256,
)

gemma_300m_config = Config(
    width=1024,
    depth=18,
    mlp_dim=4096,
    num_heads=8,
    num_kv_heads=1,
    head_dim=256,
)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps)
        normed = normed * (1 + self.scale)
        
        return normed


class Embedder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        
    def encode(self, x):
        x = self.input_embedding(x)
        x = x * math.sqrt(self.embed_dim)
        
        return x
    
    def decode(self, x):
        return F.linear(x, self.input_embedding.weight)


def apply_rope(x, positions, max_wavelength=10000):
    """Apply RoPE to input tensor x with given positions."""
    head_dim = x.shape[-1]
    freq_exponents = (2.0 / head_dim) * torch.arange(head_dim // 2, dtype=torch.float32, device=x.device)
    timescale = max_wavelength ** freq_exponents
    
    radians = positions.unsqueeze(-1).float() / timescale.unsqueeze(0).unsqueeze(0)  # [B, L, d//2]
    radians = radians.unsqueeze(-2)  # [B, L, 1, d//2]
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)
    
    return rotated.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, configs: List[Config]):
        super().__init__()
        self.configs = configs
        
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == configs[0].head_dim for config in configs)
        assert all(config.num_heads == configs[0].num_heads for config in configs)
        assert all(config.num_kv_heads == configs[0].num_kv_heads for config in configs)
        
        self.head_dim = configs[0].head_dim
        self.num_heads = configs[0].num_heads
        self.num_kv_heads = configs[0].num_kv_heads
        
        # Create projection layers for each expert
        self.q_projections = nn.ModuleList()
        self.k_projections = nn.ModuleList()
        self.v_projections = nn.ModuleList()
        self.out_projections = nn.ModuleList()
        
        for config in configs:
            self.q_projections.append(nn.Linear(config.width, config.num_heads * config.head_dim, bias=False))
            self.k_projections.append(nn.Linear(config.width, config.num_kv_heads * config.head_dim, bias=False))
            self.v_projections.append(nn.Linear(config.width, config.num_kv_heads * config.head_dim, bias=False))
            self.out_projections.append(nn.Linear(config.num_heads * config.head_dim, config.width, bias=False))
    
    def forward(self, xs, positions, attn_mask, kv_cache):
        qs, ks, vs = [], [], []
        
        for i, x in enumerate(xs):
            if x is None:
                continue

            batch_size, seq_len, _ = x.shape

            q = self.q_projections[i](x)  # [B, T, N*H]
            k = self.k_projections[i](x)  # [B, T, K*H]
            v = self.v_projections[i](x)  # [B, T, K*H]
            
            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            
            qs.append(q)
            ks.append(k)
            vs.append(v)
        
        # Concatenate all Q, K, V along sequence dimension
        q = torch.cat(qs, dim=1)  # [B, total_seq, N, H]
        k = torch.cat(ks, dim=1)  # [B, total_seq, K, H]
        v = torch.cat(vs, dim=1)  # [B, total_seq, K, H]
        
        # Apply RoPE
        q = apply_rope(q, positions)
        q = q * (self.head_dim ** -0.5)
        k = apply_rope(k, positions)
        
        # Handle KV cache
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = torch.cat([cache_k, k], dim=1)
            v = torch.cat([cache_v, v], dim=1)
        
        k_cache = k  # [B, kv_seq_len, num_kv_heads, H]
        v_cache = v  # [B, kv_seq_len, num_kv_heads, H]
        
        # Grouped Query Attention
        batch_size, q_len, num_heads, head_dim = q.shape
        _, kv_len, num_kv_heads, _ = k.shape
        
        # Expand K, V to match number of query heads
        group_size = num_heads // num_kv_heads
        k = k.unsqueeze(2).expand(-1, -1, group_size, -1, -1).reshape(batch_size, kv_len, num_heads, head_dim)
        v = v.unsqueeze(2).expand(-1, -1, group_size, -1, -1).reshape(batch_size, kv_len, num_heads, head_dim)
        
        # Compute attention scores
        q = q.transpose(1, 2)  # [B, N, T, H]
        k = k.transpose(1, 2)  # [B, N, S, H]
        v = v.transpose(1, 2)  # [B, N, S, H]

        scores = torch.matmul(q, k.transpose(-2, -1))  # [B, N, T, S]
        big_neg = torch.finfo(scores.dtype).min
        scores = torch.where(attn_mask.unsqueeze(1), scores, big_neg)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, N, T, H]
        out = out.transpose(1, 2)  # [B, T, N, H]
        out = out.reshape(batch_size, q_len, num_heads * head_dim)
        
        # Split output back to experts
        outputs = []
        start = 0
        for i, x in enumerate(xs):
            if x is not None:
                end = start + x.shape[1]
                expert_out = self.out_projections[i](out[:, start:end])
                outputs.append(expert_out)
                start = end
            else:
                outputs.append(None)
        
        return outputs, (k_cache, v_cache)


class FeedForward(nn.Module):
    def __init__(self, features, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(features, hidden_dim, bias=False)
        self.up_proj = nn.Linear(features, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, features, bias=False)
    
    def forward(self, x):
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Block(nn.Module):
    def __init__(self, configs: List[Config]):
        super().__init__()
        self.configs = configs
        
        self.attention = Attention(configs)
        self.feed_forwards = nn.ModuleList([
            FeedForward(config.width, config.mlp_dim) for config in configs
        ])
        
        self.pre_attn_norms = nn.ModuleList([
            RMSNorm(config.width) for config in configs
        ])

        self.pre_ffn_norms = nn.ModuleList([
            RMSNorm(config.width) for config in configs
        ])
    
    def forward(self, xs, kv_cache, positions, attn_mask):
        # Pre-attention normalization
        pre_attn = []
        for i, x in enumerate(xs):
            if x is not None:
                x = self.pre_attn_norms[i](x)
            pre_attn.append(x)
        
        # Attention
        post_attn, kv_cache = self.attention(pre_attn, positions, attn_mask, kv_cache)
        
        xs = [x + y if x is not None and y is not None else None for x, y in zip(xs, post_attn)]
        
        # Pre-FFN normalization and FFN
        ffn_outputs = []
        for i, x in enumerate(xs):
            if x is not None:
                normed = self.pre_ffn_norms[i](x)
                ffn_out = self.feed_forwards[i](normed)
                ffn_outputs.append(ffn_out)
            else:
                ffn_outputs.append(None)
        
        # Residual connection
        xs = [x + y if x is not None and y is not None else None for x, y in zip(xs, ffn_outputs)]
        
        return xs, kv_cache


class Gemma(nn.Module):
    def __init__(self, configs=(gemma_2b_config, gemma_300m_config)):
        super().__init__()
        self.configs = configs
        
        assert all(config.depth == configs[0].depth for config in configs)
        
        self.embedder = Embedder(PALIGEMMA_VOCAB_SIZE, configs[0].width)
        
        self.layers = nn.ModuleList([
            Block(configs) for _ in range(configs[0].depth)
        ])
        
        self.final_norms = nn.ModuleList([
            RMSNorm(config.width) for config in configs
        ])
    
    def embed(self, tokens):
        return self.embedder.encode(tokens)
    
    def forward(self, embedded, positions, mask, kv_cache=None):
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            if kv_cache is not None:
                layer_kv_cache = (kv_cache[0][i], kv_cache[1][i])
            else:
                layer_kv_cache = None
            
            embedded, new_layer_kv_cache = layer(embedded, layer_kv_cache, positions, mask)
            new_kv_cache.append(new_layer_kv_cache)
        
        outputs = []
        for i, (e, norm) in enumerate(zip(embedded, self.final_norms)):
            if e is not None:
                outputs.append(norm(e))
            else:
                outputs.append(None)

        all_k = [layer_cache[0] for layer_cache in new_kv_cache]
        all_v = [layer_cache[1] for layer_cache in new_kv_cache]
        stacked_k = torch.stack(all_k, dim=0)
        stacked_v = torch.stack(all_v, dim=0)
        
        return outputs, (stacked_k, stacked_v)
