import torch
import torch.nn as nn
from typing import Dict

from data_processing.utils import preprocess_observation
from .gemma import Gemma, gemma_300m_config
from .head import VoxelDecoder
from .keypoints import TrackEncoder
from .siglip import SigLIP


action_dim = 32
action_horizon = 50
max_token_len = 48
action_expert_config = gemma_300m_config


def make_attn_mask(input_mask, mask_ar):
    # input_mask: [B, N], mask_ar: [N]
    mask_ar = mask_ar.unsqueeze(0).expand(input_mask.shape[0], -1)  # [B, N]
    cumsum = torch.cumsum(mask_ar, dim=1)  # [B, N]
    # [B, 1, N] <= [B, N, 1]
    attn_mask = cumsum.unsqueeze(1) <= cumsum.unsqueeze(2)
    valid_mask = input_mask.unsqueeze(1) * input_mask.unsqueeze(2)
    return torch.logical_and(attn_mask, valid_mask)


def posemb_sincos(pos, embedding_dim, min_period, max_period):
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, device=pos.device)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = torch.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * torch.pi,
    )

    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)


def get_1d_sincos_pos_embed(embed_dim, pos, base=32):
    assert embed_dim % 2 == 0
    
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / base**omega  # (D/2,)
    
    pos = pos.reshape(-1)  # (L,)
    out = torch.einsum('m,d->md', pos, omega)  # (L, D/2), 外积
    
    emb_sin = torch.sin(out)  # (L, D/2)
    emb_cos = torch.cos(out)  # (L, D/2)
    emb = torch.concatenate([emb_sin, emb_cos], axis=1)  # (L, D)

    return emb


def get_3d_sincos_pos_embed(grid_x, grid_y, grid_z, embed_dim=2048):
    dim_x, dim_y, dim_z = 800, 800, 448
    
    pos_x = torch.arange(grid_x, dtype=torch.float32)
    pos_y = torch.arange(grid_y, dtype=torch.float32)
    pos_z = torch.arange(grid_z, dtype=torch.float32)
    
    emb_x = get_1d_sincos_pos_embed(dim_x, pos_x)
    emb_y = get_1d_sincos_pos_embed(dim_y, pos_y)
    emb_z = get_1d_sincos_pos_embed(dim_z, pos_z)
    
    pos_embed = torch.zeros((grid_x * grid_y * grid_z, embed_dim))
    
    idx = 0
    for x in range(grid_x):
        for y in range(grid_y):
            for z in range(grid_z):
                pos_embed[idx] = torch.concatenate([emb_x[x], emb_y[y], emb_z[z]])
                idx += 1
    
    return pos_embed


class GeoPredict(nn.Module):
    def __init__(self, embed_dtype=torch.bfloat16):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_token_len = max_token_len
        self.embed_dtype = embed_dtype

        self.llm = Gemma()
        self.img = SigLIP()
        self.state_proj = nn.Linear(action_dim, action_expert_config.width)
        self.action_in_proj = nn.Linear(action_dim, action_expert_config.width)
        self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
        self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, action_dim)

        self.future_pos = get_1d_sincos_pos_embed(2048, torch.arange(50, dtype=torch.float32), base=100).to(self.embed_dtype)

        # keypoints
        self.joint_num, self.embed_dims = 8, 2048
        self.keypoint_encoder = TrackEncoder()
        self.keypoint_embedding = nn.Embedding(self.joint_num, self.embed_dims)
        self.keypoint_out_proj = nn.Linear(self.embed_dims, 3)

        # depth rendering
        self.grid_x, self.grid_y, self.grid_z, self.embed_dims = 8, 8, 5, 2048
        self.spatial_embedding = nn.Embedding(self.grid_x * self.grid_y * self.grid_z, self.embed_dims)
        self.spatial_pos = get_3d_sincos_pos_embed(self.grid_x, self.grid_y, self.grid_z, self.embed_dims).to(self.embed_dtype)
        self.spatial_num = self.grid_x * self.grid_y * self.grid_z

        self.gs_decoder = VoxelDecoder()
        self.refine_gs_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 64*14)
        )

    def embed_prefix(self, obs: Dict):
        input_mask = []
        ar_mask = []
        tokens = []

        # embed images
        for name in obs["images"]:
            image_tokens = self.img(obs["images"][name])  # [b, h*w, d]

            tokens.append(image_tokens)
            input_mask.append(
                obs["image_masks"][name].unsqueeze(1).expand(-1, image_tokens.shape[1])
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs["tokenized_prompt"] is not None:
            tokenized_inputs = self.llm.embed(obs["tokenized_prompt"])  # [b, t, d]
            tokens.append(tokenized_inputs)
            input_mask.append(obs["tokenized_prompt_mask"])
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]

        # add history keypoint tokens
        device = self.keypoint_embedding.weight.device
        current_batch_size = tokens[0].shape[0]

        his_keypoint_token = self.keypoint_encoder(obs["his_kpts"], obs["his_len"])  # [b, j, d]
        tokens.append(his_keypoint_token)
        ar_mask += [True] + ([False] * (his_keypoint_token.shape[1] - 1))
        input_mask.append(torch.ones(his_keypoint_token.shape[:2], dtype=torch.bool).to(device))

        # add keypoint and spatial query tokens
        joint_token = self.keypoint_embedding.weight.unsqueeze(0).repeat(current_batch_size, 1, 1)
        tokens.append(joint_token)
        ar_mask += [True] + ([False] * (joint_token.shape[1] - 1))
        input_mask.append(torch.ones(joint_token.shape[:2], dtype=torch.bool).to(device))

        spatial_token = (self.spatial_embedding.weight + self.spatial_pos.to(device)).unsqueeze(0).repeat(current_batch_size, 1, 1)
        tokens.append(spatial_token)
        ar_mask += [False] * spatial_token.shape[1]
        input_mask.append(torch.ones(spatial_token.shape[:2], dtype=torch.bool).to(device))

        # concate all tokens and masks
        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1)
        ar_mask = torch.tensor(ar_mask).to(tokens.device)

        # [b, h*w + t, d], [b, h*w + t], [h*w + t]
        return tokens, input_mask, ar_mask

    def embed_suffix(self, obs, noisy_actions, timestep):
        input_mask = []
        ar_mask = []
        tokens = []

        # add a single state token
        state_token = self.state_proj(obs["state"])[:, None, :]  # [b, 1, d]
        tokens.append(state_token)
        input_mask.append(torch.ones((obs["state"].shape[0], 1), dtype=torch.bool))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0).to(self.embed_dtype)  # [b, d]
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(noisy_actions.to(self.embed_dtype))  # [b, ah, d]
        time_tokens = time_emb.unsqueeze(1).repeat(1, self.action_horizon, 1)  # [b, ah, d]
        action_time_tokens = torch.cat([action_tokens, time_tokens], dim=-1)  # [b, ah, 2d]
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)  # [b, ah, d]
        action_time_tokens = torch.nn.functional.silu(action_time_tokens)  # [b, ah, d]
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)  # [b, ah, d]
        tokens.append(action_time_tokens)
        input_mask.append(torch.ones(action_time_tokens.shape[:2], dtype=torch.bool))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1).to(tokens.device)
        ar_mask = torch.tensor(ar_mask).to(tokens.device)

        # [b, 1 + ah, d], [b, 1 + ah], [1 + ah]
        return tokens, input_mask, ar_mask

    def sample_actions(self, observation, num_steps=10):
        observation = preprocess_observation(observation, train=self.training)

        dt = -1.0 / num_steps
        batch_size = observation["state"].shape[0]
        device = observation["state"].device
        x_t = torch.randn((batch_size, self.action_horizon, self.action_dim), device=device)
        time = torch.tensor(1.0, device=device)

        # First fill KV cache with prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = torch.cumsum(prefix_mask, dim=1) - 1
        (prefix_out, suffix_out), kv_cache = self.llm([prefix_tokens, None], positions=positions, mask=prefix_attn_mask)

        while time >= -dt / 2:
            time_batch = time.expand(batch_size)
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time_batch)
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)  # [b, s, s]
            prefix_attn_mask_expanded = prefix_mask.unsqueeze(1).expand(-1, suffix_tokens.shape[1], -1)  # [b, s, p]
            full_attn_mask = torch.cat([prefix_attn_mask_expanded, suffix_attn_mask], dim=-1)  # [b, s, p + s]
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # [b, p] -> [b, 1] + [b, s] -1 -> [b, s]
            positions = torch.sum(prefix_mask, dim=-1).unsqueeze(1) + torch.cumsum(suffix_mask, dim=-1) - 1
            
            # Forward pass with KV cache
            (prefix_out, suffix_out), _ = self.llm(
                [None, suffix_tokens], positions=positions, mask=full_attn_mask, kv_cache=kv_cache)
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])  # [b, ah, ad]
            
            x_t = x_t + dt * v_t
            time = time + dt

        return x_t
