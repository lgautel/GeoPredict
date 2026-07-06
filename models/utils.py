import torch


def get_voxel_means_torch(point_cloud_range, voxel_dims, device='cpu'):
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    nx, ny, nz = voxel_dims

    voxel_size_x = (x_max - x_min) / nx
    voxel_size_y = (y_max - y_min) / ny
    voxel_size_z = (z_max - z_min) / nz
    
    pcr = torch.tensor(point_cloud_range, dtype=torch.float32, device=device)
    voxel_size = torch.tensor([voxel_size_x, voxel_size_y, voxel_size_z], dtype=torch.float32, device=device)

    x_centers = torch.linspace(pcr[0] + voxel_size[0] / 2, pcr[3] - voxel_size[0] / 2, nx, device=device)
    y_centers = torch.linspace(pcr[1] + voxel_size[1] / 2, pcr[4] - voxel_size[1] / 2, ny, device=device)
    z_centers = torch.linspace(pcr[2] + voxel_size[2] / 2, pcr[5] - voxel_size[2] / 2, nz, device=device)

    xx, yy, zz = torch.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

    voxel_means = torch.stack([xx, yy, zz], dim=-1)  # Shape: (nx, ny, nz, 3)

    return voxel_means


def lift_to_3d(depth, intrinsics, extrinsics):
    B, H, W = depth.shape
    device = depth.device

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    pixels_homogeneous = torch.stack(
        (x_coords.flatten(), y_coords.flatten(), torch.ones(H * W, device=device)),
        dim=1
    )  # (H*W, 3)

    #  pixel -> camera
    intrinsics_inv = torch.inverse(intrinsics)
    #  (B, 3, 3) @ (B, 3, H*W) -> (B, 3, H*W)
    cam_coords_normalized = intrinsics_inv @ pixels_homogeneous.T.unsqueeze(0)
    depth_flat = depth.view(B, 1, H * W)
    cam_coords = depth_flat * cam_coords_normalized
    
    # camera -> world
    ones = torch.ones((B, 1, H * W), device=device)
    cam_coords_homogeneous = torch.cat((cam_coords, ones), dim=1)  # (B, 4, H*W)
    world_coords_homogeneous = extrinsics @ cam_coords_homogeneous  # (B, 4, H*W)
    points_3d_world = world_coords_homogeneous[:, :3, :]  # (B, 3, H*W)

    points_3d_world = points_3d_world.permute(0, 2, 1).view(B, H, W, 3)
    
    return points_3d_world


def get_voxel_indices_torch(points, point_cloud_range, voxel_dims, device='cpu', 
                            expand_neighborhood=False, neighborhood_size=3):
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    nx, ny, nz = voxel_dims
    
    voxel_size_x = (x_max - x_min) / nx
    voxel_size_y = (y_max - y_min) / ny
    voxel_size_z = (z_max - z_min) / nz
    
    pcr_min = torch.tensor([x_min, y_min, z_min], dtype=torch.float32, device=device)
    pcr_max = torch.tensor([x_max, y_max, z_max], dtype=torch.float32, device=device)
    voxel_size = torch.tensor([voxel_size_x, voxel_size_y, voxel_size_z], dtype=torch.float32, device=device)
    
    voxel_indices = torch.floor((points - pcr_min) / voxel_size).long()
    valid_mask = (points >= pcr_min).all(dim=1) & (points < pcr_max).all(dim=1)
    voxel_dims_tensor = torch.tensor([nx, ny, nz], dtype=torch.long, device=device)

    if not expand_neighborhood:
        return voxel_indices, valid_mask
    
    N = voxel_indices.shape[0]
    
    offset = neighborhood_size // 2
    offsets = torch.arange(-offset, offset + 1, device=device)
    
    dx, dy, dz = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
    offset_grid = torch.stack([dx.flatten(), dy.flatten(), dz.flatten()], dim=-1)
    
    neighborhood_indices = voxel_indices.unsqueeze(1) + offset_grid.unsqueeze(0)
    
    neighborhood_valid = (neighborhood_indices >= 0).all(dim=2) & \
                        (neighborhood_indices < voxel_dims_tensor).all(dim=2)
    
    return voxel_indices, valid_mask, neighborhood_indices, neighborhood_valid
