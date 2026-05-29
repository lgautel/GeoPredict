import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PointPatchEmbedding(nn.Module):
    def __init__(self, patch_size=4, in_dim=3, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv1d(in_dim, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        
    def forward(self, points, lengths):
        # points: (batch_size, time_len, num_points, in_dim)
        # lengths: (batch_size,)
        batch_size, _, num_points, in_dim = points.shape
        patch_size = self.patch_size
        
        processed_points = []
        updated_lengths = []
        for i in range(batch_size):
            actual_len = lengths[i].item()
            batch_points = points[i, :actual_len] 
            if actual_len % patch_size != 0:
                pad_len = patch_size - (actual_len % patch_size)
                padding = batch_points[-1:].repeat(pad_len, 1, 1)
                batch_points = torch.cat([batch_points, padding], dim=0)
            
            processed_points.append(batch_points)
            updated_lengths.append(batch_points.size(0))
        
        max_padded_len = max(len(bp) for bp in processed_points)
        final_points = torch.zeros(batch_size, max_padded_len, num_points, in_dim, 
                                   dtype=points.dtype, device=points.device)
        for i, batch_points in enumerate(processed_points):
            final_points[i, :batch_points.size(0)] = batch_points
        
        points = final_points
        lengths = torch.tensor(updated_lengths, dtype=torch.long, device=points.device)
        
        points_reshaped = rearrange(points, 'b t n c -> (b n) c t')
        patches = self.conv(points_reshaped)
        patches = rearrange(patches, '(b n) c t -> b t n c', b=batch_size, n=num_points)
        patch_lengths = lengths // patch_size

        # patches: (batch_size, num_patches, num_points, embed_dim)
        # patch_lengths: (batch_size,), different for every sample in the batch
        return patches, patch_lengths


class TimeEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=10000, embedding_type='sinusoidal'):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.embedding_type = embedding_type
        self.register_buffer('pos_embedding', self._create_sinusoidal_embeddings(max_seq_len, dim))
    
    def _create_sinusoidal_embeddings(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
        
    def forward(self, positions):
        return self.pos_embedding[positions]


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads=8, dropout=0.1, time_embedding_type='sinusoidal', max_seq_len=10000):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        
        assert query_dim % num_heads == 0
        self.q_linear, self.k_linear, self.v_linear = nn.Linear(query_dim, query_dim), nn.Linear(key_dim, query_dim), nn.Linear(key_dim, query_dim)
        self.out_linear = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        self.key_time_embedding = TimeEmbedding(key_dim, max_seq_len, time_embedding_type)
        
    def forward(self, query, key, value, mask, key_positions):
        bs, q_len, k_len = query.size(0), query.size(1), key.size(1)
        
        key_pos_emb = self.key_time_embedding(key_positions)  # (k_len, key_dim)
        key_pos_emb = key_pos_emb.unsqueeze(0).expand(bs, -1, -1)  # (bs, k_len, key_dim)
        key = key + key_pos_emb
        
        Q = self.q_linear(query).view(bs, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, q_len, head_dim)  
        K = self.k_linear(key).view(bs, k_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, k_len, head_dim)
        V = self.v_linear(value).view(bs, k_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, k_len, head_dim)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (bs, num_heads, q_len, k_len)
        if mask is not None:  # mask: (bs, 1, k_len)
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, q_len, -1)
            scores = scores.masked_fill(mask==0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # (bs, q_len, query_dim)
        out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(bs, q_len, self.num_heads * self.head_dim)

        return self.out_linear(out)


class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads=8, ff_dim=1024, dropout=0.1, 
                 time_embedding_type='sinusoidal', max_seq_len=10000):
        super().__init__()
        self.norm_cross = nn.LayerNorm(query_dim)
        self.cross_attn = MultiHeadAttention(
            query_dim, key_dim, num_heads, dropout,
            time_embedding_type=time_embedding_type, max_seq_len=max_seq_len)
        
        self.norm_ffn = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, ff_dim),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(ff_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, queries, inputs, input_mask, input_positions):
        # queries: (batch_size, num_queries, query_dim)
        # inputs: (batch_size, seq_len, key_dim)
        # input_mask: (batch_size, seq_len)
        # input_positions: (seq_len,)
        input_mask = input_mask.unsqueeze(1) 
        cross_attn_out = self.cross_attn(
            self.norm_cross(queries), 
            inputs, 
            inputs, 
            mask=input_mask, 
            key_positions=input_positions
        )  # (bs, q_len, query_dim)
        queries = queries + cross_attn_out 

        ffn_out = self.ffn(self.norm_ffn(queries)) 
        queries = queries + ffn_out 
        
        return queries


class TrackEncoder(nn.Module):
    def __init__(
        self,
        input_dim=3,
        output_dim=2048,
        patch_size=4,
        embed_dim=256,
        query_dim=512,
        num_queries=1,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
        max_seq_len=1000
    ):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, query_dim))
        nn.init.xavier_uniform_(self.queries)
        
        self.point_patch_embed = PointPatchEmbedding(
            patch_size=patch_size, in_dim=input_dim, embed_dim=embed_dim)
        self.cross_attention_block = CrossAttentionBlock(
            query_dim, embed_dim, num_heads, ff_dim, dropout, max_seq_len=max_seq_len//patch_size)
        self.linear_transform = nn.Sequential(
            nn.Linear(query_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, query_dim)
        )  
        self.final_norm = nn.LayerNorm(query_dim)

        self.track_fusion_layer = nn.Linear(query_dim, output_dim)
        
    def forward(self, points, lengths):
        batch_size = points.size(0)
        # patches: (batch_size, num_patches, num_points, embed_dim)
        # patch_lengths: (batch_size,), different for every sample in the batch
        patches, patch_lengths = self.point_patch_embed(points, lengths)
        num_patches, num_points = patches.size(1), patches.size(2)
        input_positions = torch.arange(num_patches).to(points.device)
        
        all_point_outputs = []
        for point_idx in range(num_points):
            point_patches = patches[:, :, point_idx, :]
            point_queries = self.queries.expand(batch_size, -1, -1)
            point_mask = torch.arange(num_patches, device=points.device)[None, :] < patch_lengths[:, None]
            
            point_queries = self.cross_attention_block(
                point_queries, 
                point_patches, 
                point_mask, 
                input_positions
            )  # (bs, q_len, query_dim)

            point_queries = self.linear_transform(point_queries)
            all_point_outputs.append(point_queries)

        # (bs, num_points, q_len, query_dim)
        output = torch.stack(all_point_outputs, dim=1)
        output = self.final_norm(output)
        output = output.reshape(batch_size, -1, output.size(-1))  # (bs, num_points * q_len, query_dim)
        output = self.track_fusion_layer(output)  # (bs, num_points * q_len, output_dim)

        return output
