import torch
try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    _HAS_DIFF_GAUSSIAN = True
except ImportError:
    GaussianRasterizationSettings = None
    GaussianRasterizer = None
    _HAS_DIFF_GAUSSIAN = False


def getProjectionMatrixK(K, H, W, znear, zfar, device="cuda"):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]

    P = torch.zeros((4, 4), device=device)

    z_sign = 1.0

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -1 * z_sign * 2 * zfar * znear / (zfar - znear)
    P[3, 2] = z_sign

    return P


def get_cam_info_gaussian_v2(c2w, K, H, W, znear, zfar):
    world_view_transform = torch.inverse(c2w.float())  # w2c

    world_view_transform = world_view_transform.transpose(0, 1).cuda().float()
    projection_matrix = (
        getProjectionMatrixK(K, H, W, znear, zfar)
        .transpose(0, 1)
        .cuda()
    ).float()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0).float()
    camera_center = world_view_transform.inverse()[3, :3]

    return world_view_transform, full_proj_transform, camera_center


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


class GaussianRenderer:
    '''
    In origin version, get_cam_info_gaussian only supports cam with optical center in image center.
    This version supports arbitrary camera setting.
    '''
    def __init__(
        self, 
        resolution: list = [512, 512],
        znear: float = 0.1,
        zfar: float = 100.0, 
        renderer_type: str = "vanilla", # only support "vanilla"
    ):
        self.renderer_type = renderer_type
        self.resolution = resolution
        self.znear = znear
        self.zfar = zfar
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def render(
        self, 
        gaussians, 
        c2w,
        fovx=None,
        fovy=None,
        bg_color=None, 
        scale_modifier=1.,
        K=None,
        H=None,
        W=None
    ):
        if not _HAS_DIFF_GAUSSIAN:
            raise ImportError(
                "diff_gaussian_rasterization is required for depth rendering. "
                "Install it or disable depth loss."
            )
        # at least one of fovx and fovy is not none
        assert fovx is not None or fovy is not None
        if fovx is None:
            fovx = fovy
        if fovy is None:
            fovy = fovx

        device = gaussians.device
        B, V = c2w.shape[:2]

        # loop of loop...
        images = []
        alphas = []
        depths = []
        for b in range(B):
            means3D = gaussians[b, :, 0:3].contiguous().float()  # [160000, 3]
            rgbs = gaussians[b, :, 3:6].contiguous().float()  # [160000, 3]
            opacity = gaussians[b, :, 6:7].contiguous().float()  # [160000, 1]
            rotations = gaussians[b, :, 7:11].contiguous().float()  # [160000, 4]
            scales = gaussians[b, :, 11:].contiguous().float()  # [160000, 3]
            means2D = torch.zeros_like(means3D, dtype=means3D.dtype, device=device)  # [160000, 3]

            for v in range(V):
                fovx_ = fovx[b, v].clone()
                fovy_ = fovy[b, v].clone()
                c2w_ = c2w[b, v].clone()
                K_ = K[b, v].clone()
                H_ = H[b, v]
                W_ = W[b, v]
                w2c, proj, cam_p = get_cam_info_gaussian_v2(
                    c2w=c2w_, K=K_, H=H_, W=W_, znear=self.znear, zfar=self.zfar
                )
                # render novel views
                tan_half_fovx = torch.tan(fovx_ * 0.5)
                tan_half_fovy = torch.tan(fovy_ * 0.5)

                if self.renderer_type == "vanilla":
                    raster_settings = GaussianRasterizationSettings(
                        image_height=self.resolution[0],
                        image_width=self.resolution[1],
                        tanfovx=tan_half_fovx,
                        tanfovy=tan_half_fovy,
                        bg=self.bg_color if bg_color is None else bg_color,
                        scale_modifier=scale_modifier,
                        viewmatrix=w2c,
                        projmatrix=proj,
                        sh_degree=0,
                        campos=cam_p,
                        prefiltered=False,
                        debug=False,
                    )
                    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                else:
                    raise NotImplementedError

                # Rasterize visible Gaussians to image, obtain their radii (on screen).
                if self.renderer_type == "vanilla":
                    rendered_image, _, rendered_depth, rendered_alpha = rasterizer(
                        means3D=means3D,
                        means2D=means2D,
                        shs=None,
                        colors_precomp=rgbs,
                        opacities=opacity,
                        scales=scales,
                        rotations=rotations,
                        cov3D_precomp=None,
                    )
                else:
                    raise NotImplementedError

                rendered_image = torch.clamp(rendered_image, min=0.0, max=1.0)
                images.append(rendered_image)
                alphas.append(rendered_alpha)
                depths.append(rendered_depth)

        images = torch.stack(images, dim=0).view(B, V, 3, self.resolution[0], self.resolution[1])
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, self.resolution[0], self.resolution[1])
        depths = torch.stack(depths, dim=0).view(B, V, 1, self.resolution[0], self.resolution[1])

        return {
            "image": images,  # [B, V, 3, H, W]
            "alpha": alphas,  # [B, V, 1, H, W]
            "depth": depths,  # [B, V, 1, H, W]
        }
