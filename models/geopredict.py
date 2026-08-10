import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from data_processing.utils import preprocess_observation
from .gaussian import GaussianRenderer
from .gemma import Gemma, gemma_300m_config
from .head import VoxelDecoder
from .keypoints import TrackEncoder
from .siglip import SigLIP
from .utils import get_voxel_means_torch, lift_to_3d, get_voxel_indices_torch


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


def move_to_type(data, data_type):
    if isinstance(data, torch.Tensor) and data.dtype == torch.float32:
        return data.to(data_type)
    elif isinstance(data, dict):
        return {key: move_to_type(value, data_type) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_type(item, data_type) for item in data)
    else:
        return data


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
    def __init__(self, embed_dtype=torch.bfloat16, joint_num=8, use_depth_loss=True):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_token_len = max_token_len
        self.embed_dtype = embed_dtype
        self.use_depth_loss = use_depth_loss

        self.llm = Gemma()
        self.img = SigLIP()
        self.state_proj = nn.Linear(action_dim, action_expert_config.width)
        self.action_in_proj = nn.Linear(action_dim, action_expert_config.width)
        self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
        self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, action_dim)

        self.future_pos = get_1d_sincos_pos_embed(2048, torch.arange(50, dtype=torch.float32), base=100).to(self.embed_dtype)

        # keypoints
        self.joint_num, self.embed_dims = joint_num, 2048
        self.keypoint_encoder = TrackEncoder()
        self.keypoint_embedding = nn.Embedding(self.joint_num, self.embed_dims)
        self.keypoint_out_proj = nn.Linear(self.embed_dims, 3)

        # depth rendering
        self.grid_x, self.grid_y, self.grid_z, self.embed_dims = 8, 8, 5, 2048
        self.spatial_embedding = nn.Embedding(self.grid_x * self.grid_y * self.grid_z, self.embed_dims)
        self.spatial_pos = get_3d_sincos_pos_embed(self.grid_x, self.grid_y, self.grid_z, self.embed_dims).to(self.embed_dtype)
        self.spatial_num = self.grid_x * self.grid_y * self.grid_z

        self.offset_act = lambda x: F.tanh(x) * 0.04
        self.opt_act = torch.sigmoid
        self.scale_act = lambda x: F.softplus(x, beta=20.0)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = torch.sigmoid
        self.gs_decoder = VoxelDecoder()
        self.renderer = (
            GaussianRenderer(resolution=[224, 224], znear=0.01, zfar=10.0)
            if use_depth_loss else None
        )
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
    
    def process_guassian_voxel(self, gaussians, means):
        B = gaussians.shape[0]
        gaussians = gaussians.permute(0, 2, 3, 4, 1)
        gaussians = gaussians.reshape(B, -1, 4, 14)
        means = means[:, :, :, :, None, :].repeat(1, 1, 1, 1, 4, 1)
        means = means.reshape(B, -1, 4, 3)
        
        gaussians = gaussians.reshape(B, -1, 14)
        means = means.reshape(B, -1, 3)
        
        offsets = gaussians[..., :3]
        opacities = self.opt_act(gaussians[..., 3:4])
        scales = self.scale_act(gaussians[..., 4:7])
        scales = torch.clamp_max(scales, 0.04)
        rotations = self.rot_act(gaussians[..., 7:11])
        rgbs = self.rgb_act(gaussians[..., 11:14])
        means = means + offsets
        gaussians = torch.cat([means, rgbs, opacities, rotations, scales], dim=-1)
        
        return gaussians

    def compute_loss(self, data):
        loss_dict = {}
        acc_dict = {}

        step = data['step']
        future_steps = data['future_steps']
        actions = data['actions']
        kpt_t = data['kpt_t']
        future_kpts = data['future_kpts']
        depths_t = data['depths_t']
        depths_future = data['depths_future']
        cam_infos = data['cam_infos']

        observation = preprocess_observation(data, train=self.training)
        observation = move_to_type(observation, self.embed_dtype)
        actions = move_to_type(actions, self.embed_dtype)
        kpt_t = move_to_type(kpt_t, self.embed_dtype)
        future_kpts = move_to_type(future_kpts, self.embed_dtype)

        batch_shape = actions.shape[:-2]
        noise = torch.randn_like(actions).to(actions.device)  # [b, ah, ad]
        time = torch.distributions.Beta(1.5, 1).sample(batch_shape).to(actions.device) * 0.999 + 0.001
        time_expanded = time[..., None, None]  # [b, 1, 1]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)  # [b, h*w + t + 1 + ah]
        ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)  # [h*w + t + 1 + ah]
        attn_mask = make_attn_mask(input_mask, ar_mask)  # [b, n, n]
        positions = torch.cumsum(input_mask, dim=1) - 1  # [b, n]
        
        # Forward pass through LLM
        (prefix_out, suffix_out), _ = self.llm(
            [prefix_tokens, suffix_tokens], positions=positions, mask=attn_mask)  # [b, 1 + ah, d]

        # action loss
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])  # [b, ah, ad]
        action_loss = torch.square(v_t - u_t).mean()
        losses = action_loss
        loss_dict['action_loss'] = action_loss.item()

        # current keypoint loss
        keypoint_token = prefix_out[:, -self.spatial_num - self.joint_num:-self.spatial_num]  # [b, j, d]
        pred_kpt = self.keypoint_out_proj(keypoint_token)  # [b, j, 3]
        kpt_loss = torch.square(pred_kpt - kpt_t).mean()
        losses += kpt_loss
        loss_dict['current_keypoint_loss'] = kpt_loss.item()

        # future keypoint loss
        B = future_kpts.shape[0]
        relative_pos = future_steps - step.unsqueeze(1)  # [B, action_horizon]
        valid_mask = (relative_pos > 0) & (relative_pos <= 50)  # [B, action_horizon]
        valid_pos = torch.clamp(relative_pos - 1, 0, 49)
        pos_embeddings = self.future_pos.to(actions.device)[valid_pos].unsqueeze(2)  # [B, action_horizon, 1, embed_dim]

        future_kpt_tokens = keypoint_token.unsqueeze(1).repeat(1, self.action_horizon, 1, 1)
        future_kpt_tokens = future_kpt_tokens + pos_embeddings * valid_mask.unsqueeze(2).unsqueeze(3)
        future_kpt_tokens_flat = future_kpt_tokens.reshape(B * self.action_horizon, self.joint_num, -1)
        future_kpt_pred = self.keypoint_out_proj(future_kpt_tokens_flat)  # [B*action_horizon, joint_num, 3]
        future_kpt_flat = future_kpts.reshape(B * self.action_horizon, self.joint_num, 3)

        future_kpt_loss = torch.square(future_kpt_pred - future_kpt_flat).mean()
        losses += future_kpt_loss
        loss_dict['future_keypoint_loss'] = future_kpt_loss.item()

        if not self.use_depth_loss:
            return losses, loss_dict, acc_dict

        # current depth loss
        B, H, W = depths_t['left_depth'].shape[:3]
        spatial_token = prefix_out[:, -self.spatial_num:]  # 4,320,2048
        voxel_gs, voxel_features = self.gs_decoder(spatial_token)  # [B, 14*4, 40, 40, 25], [B, 256, 40, 40, 25]
        voxel_means = get_voxel_means_torch([0.0, 0.0, 0.0, 1.6, 1.6, 1.0], [40, 40, 25], device=voxel_gs.device)
        voxel_means = voxel_means[None, :, :, :, :].repeat(B, 1, 1, 1, 1)  # [B, 40, 40, 25, 3]
        voxel_gs = self.process_guassian_voxel(voxel_gs, voxel_means)  # [B, 160000, 14]

        c2w = []
        intrinsic = []
        for _, cam_params in cam_infos.items():
            c2w.append(cam_params['outer'].inverse())
            intrinsic.append(cam_params['inner'])
        c2w = torch.stack(c2w, dim=1).to(voxel_gs.device)  # [B, num_cams, 4, 4]
        intrinsic = torch.stack(intrinsic, dim=1).to(voxel_gs.device)  # [B, num_cams, 3, 3]

        input_fxs = intrinsic[:, :, 0, 0]  # [B, num_cams]
        input_fys = intrinsic[:, :, 1, 1]  # [B, num_cams]
        input_cxs = intrinsic[:, :, 0, 2]  # [B, num_cams]
        input_cys = intrinsic[:, :, 1, 2]  # [B, num_cams]
        input_fovxs = 2 * torch.arctan(input_cxs / input_fxs)  # [B, num_cams]
        input_fovys = 2 * torch.arctan(input_cys / input_fys)  # [B, num_cams]

        num_cams = c2w.shape[1]
        H_stack = torch.tensor([H]*B, dtype=torch.int, device=voxel_gs.device)  # B
        W_stack = torch.tensor([W]*B, dtype=torch.int, device=voxel_gs.device)  # B
        H_stack = H_stack[:, None].repeat(1, num_cams)  # [B, num_cams]
        W_stack = W_stack[:, None].repeat(1, num_cams)  # [B, num_cams]

        loss_current_depth = {
            'current_loss_render_left_depth': 0.0,
            'current_loss_render_right_depth': 0.0,
        }
        acc_current_depth = {
            'current_left_depth_mae': 0.0,
            'current_right_depth_mae': 0.0,
        }
        for i in range(B):
            key_points = pred_kpt[i]
            _, _, key_voxel, key_voxel_mask = get_voxel_indices_torch(
                key_points, [0.0, 0.0, 0.0, 1.6, 1.6, 1.0], [40, 40, 25], device=voxel_gs.device, expand_neighborhood=True, neighborhood_size=3)
            key_voxel = key_voxel[key_voxel_mask]
            key_voxel_unique = torch.unique(key_voxel, dim=0)
            tmp_voxel_feature = voxel_features[i]  # [256, 40, 40, 25]
            key_voxel_feature = tmp_voxel_feature.permute(1, 2, 3, 0)[key_voxel_unique[:, 0], key_voxel_unique[:, 1], key_voxel_unique[:, 2]]
            refine_gs = self.refine_gs_mlp(key_voxel_feature).reshape(-1, 64, 14)
            
            voxel_size, gs_per_dim = 0.04, 4
            sub_voxel_size = voxel_size / gs_per_dim
            offsets_1d = torch.arange(gs_per_dim, device=voxel_gs.device, dtype=torch.float32)
            offsets_1d = (offsets_1d + 0.5) * sub_voxel_size - voxel_size / 2
            offset_x, offset_y, offset_z = torch.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing='ij')
            offsets_3d = torch.stack([offset_x.flatten(), offset_y.flatten(), offset_z.flatten()], dim=-1)  # (64, 3)
            voxel_centers = key_voxel_unique.float() * voxel_size + voxel_size / 2
            init_means = voxel_centers.unsqueeze(1) + offsets_3d.unsqueeze(0)

            offsets = refine_gs[..., :3]
            opacities = self.opt_act(refine_gs[..., 3:4])
            scales = self.scale_act(refine_gs[..., 4:7])
            scales = torch.clamp_max(scales, 0.02)
            rotations = self.rot_act(refine_gs[..., 7:11])
            rgbs = self.rgb_act(refine_gs[..., 11:14])
            means = offsets + init_means
            refine_gs = torch.cat([means, rgbs, opacities, rotations, scales], dim=-1)
            refine_gs = refine_gs.reshape(-1, 14)

            ensembel_gaussians = torch.cat([voxel_gs[i:i+1], refine_gs[None, :, :]], dim=1)
            tmp = self.renderer.render(
                gaussians=ensembel_gaussians,
                c2w=c2w[i:i+1],
                fovx=input_fovxs[i:i+1],
                fovy=input_fovys[i:i+1],
                K=intrinsic[i:i+1],
                H=H_stack[i:i+1],
                W=W_stack[i:i+1],
            )

            cam_id = 0
            for cam, cam_depths in depths_t.items():
                cam_depth = cam_depths[i]
                inner_ = intrinsic[i:i+1, cam_id, :, :]
                c2w_ = c2w[i:i+1, cam_id, :, :]
                points = lift_to_3d(cam_depth[None], inner_, c2w_)  # [1, H, W, 3]
                point_range = torch.tensor([0.0, 0.0, 0.0, 1.6, 1.6, 1.0], device=voxel_gs.device)
                mask = (points > point_range[:3][None, None, None, :]) & (points < point_range[3:][None, None, None, :])
                mask = mask.all(dim=-1)  # [1, H, W]

                render_depths = tmp['depth'][:, cam_id, 0]  # [1, H, W]
                loss_render_depth = nn.functional.smooth_l1_loss(render_depths, cam_depth[None], reduction='none')
                loss_render_depth = (loss_render_depth * mask).sum() / (mask.sum() + 1e-6)
                loss_current_depth[f'current_loss_render_{cam}'] += loss_render_depth / B
                acc_current_depth[f'current_{cam}_mae'] += torch.mean(torch.abs(render_depths[mask] - cam_depth[None][mask])).float().item() / B

                cam_id += 1
        
        for key in loss_current_depth:
            losses += loss_current_depth[key]
            loss_dict[key] = loss_current_depth[key].item()

        acc_dict.update(acc_current_depth)

        # future depth loss
        future_spatial_tokens = spatial_token.unsqueeze(1).repeat(1, self.action_horizon, 1, 1)  # [B, action_horizon, spatial_num, embed_dim]
        future_spatial_tokens = future_spatial_tokens + pos_embeddings * valid_mask.unsqueeze(2).unsqueeze(3)  # [B, action_horizon, spatial_num, embed_dim]
        future_spatial_tokens_flat = future_spatial_tokens.reshape(B * self.action_horizon, self.spatial_num, -1)  # [B*action_horizon, spatial_num, embed_dim]
        voxel_gs_future_flat, voxel_features_future = self.gs_decoder(future_spatial_tokens_flat)  # [B*action_horizon, 14*4, 40, 40, 25]
        voxel_gs_future_all = voxel_gs_future_flat.reshape(B, self.action_horizon, *voxel_gs_future_flat.shape[1:])  # [B, action_horizon, 14*4, 40, 40, 25]
        voxel_features_future = voxel_features_future.reshape(B, self.action_horizon, *voxel_features_future.shape[1:])
        
        loss_future_depth = {
            'future_loss_render_left_depth': 0.0,
            'future_loss_render_right_depth': 0.0,
        }
        acc_future_depth = {
            'future_left_depth_mae': 0.0,
            'future_right_depth_mae': 0.0,
        }
        refine_future_kpt_pred = future_kpt_pred.reshape(B, self.action_horizon, self.joint_num, 3)
        for i in range(B):
            for t in range(self.action_horizon):
                key_points = refine_future_kpt_pred[i, t]
                _, _, key_voxel, key_voxel_mask = get_voxel_indices_torch(
                    key_points, [0.0, 0.0, 0.0, 1.6, 1.6, 1.0], [40, 40, 25], device=voxel_gs.device, expand_neighborhood=True, neighborhood_size=3)
                key_voxel = key_voxel[key_voxel_mask]
                key_voxel_unique = torch.unique(key_voxel, dim=0)
                tmp_voxel_feature = voxel_features_future[i, t]  # [256, 40, 40, 25]
                key_voxel_feature = tmp_voxel_feature.permute(1, 2, 3, 0)[key_voxel_unique[:, 0], key_voxel_unique[:, 1], key_voxel_unique[:, 2]]
                refine_gs = self.refine_gs_mlp(key_voxel_feature).reshape(-1, 64, 14)
                
                voxel_size, gs_per_dim = 0.04, 4
                sub_voxel_size = voxel_size / gs_per_dim
                offsets_1d = torch.arange(gs_per_dim, device=voxel_gs.device, dtype=torch.float32)
                offsets_1d = (offsets_1d + 0.5) * sub_voxel_size - voxel_size / 2
                offset_x, offset_y, offset_z = torch.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing='ij')
                offsets_3d = torch.stack([offset_x.flatten(), offset_y.flatten(), offset_z.flatten()], dim=-1)  # (64, 3)
                voxel_centers = key_voxel_unique.float() * voxel_size + voxel_size / 2
                init_means = voxel_centers.unsqueeze(1) + offsets_3d.unsqueeze(0)

                offsets = refine_gs[..., :3]
                opacities = self.opt_act(refine_gs[..., 3:4])
                scales = self.scale_act(refine_gs[..., 4:7])
                scales = torch.clamp_max(scales, 0.02)
                rotations = self.rot_act(refine_gs[..., 7:11])
                rgbs = self.rgb_act(refine_gs[..., 11:14])
                means = offsets + init_means
                refine_gs = torch.cat([means, rgbs, opacities, rotations, scales], dim=-1)
                refine_gs = refine_gs.reshape(-1, 14)

                tmp_voxel_gs = voxel_gs_future_all[i, t]
                voxel_gs = self.process_guassian_voxel(tmp_voxel_gs[None], voxel_means[i:i+1])  # [1, 160000, 14]
                ensembel_gaussians = torch.cat([voxel_gs, refine_gs[None, :, :]], dim=1)
                tmp = self.renderer.render(
                    gaussians=ensembel_gaussians,
                    c2w=c2w[i:i+1],
                    fovx=input_fovxs[i:i+1],
                    fovy=input_fovys[i:i+1],
                    K=intrinsic[i:i+1],
                    H=H_stack[i:i+1],
                    W=W_stack[i:i+1],
                )

                cam_id = 0
                for cam, cam_depths in depths_future.items():
                    cam_depth = cam_depths[i, t]
                    inner_ = intrinsic[i:i+1, cam_id, :, :]
                    c2w_ = c2w[i:i+1, cam_id, :, :]
                    points = lift_to_3d(cam_depth[None], inner_, c2w_)  # [1, H, W, 3]
                    point_range = torch.tensor([0.0, 0.0, 0.0, 1.6, 1.6, 1.0], device=voxel_gs.device)
                    mask = (points > point_range[:3][None, None, None, :]) & (points < point_range[3:][None, None, None, :])
                    mask = mask.all(dim=-1)  # [1, H, W]

                    render_depths = tmp['depth'][:, cam_id, 0]  # [1, H, W]
                    loss_render_depth = nn.functional.smooth_l1_loss(render_depths, cam_depth[None], reduction='none')  # [B, H, W]
                    loss_render_depth = (loss_render_depth * mask).sum() / (mask.sum() + 1e-6)
                    loss_future_depth[f'future_loss_render_{cam}'] += loss_render_depth / (B * self.action_horizon)
                    acc_future_depth[f'future_{cam}_mae'] += torch.mean(torch.abs(render_depths[mask] - cam_depth[None][mask])).float().item() / (B * self.action_horizon)

                    cam_id += 1

        for key in loss_future_depth:
            losses += loss_future_depth[key]
            loss_dict[key] = loss_future_depth[key].item()

        acc_dict.update(acc_future_depth)

        return losses, loss_dict, acc_dict

    def forward(self, data):
        return self.compute_loss(data)

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
