import torch
from torch import nn

from .hdmapnet_utils.homography import IPM

from mmdet3d.models.builder import BACKBONES

from efficientnet_pytorch import EfficientNet


class ViewTransformation(nn.Module):
    def __init__(self, fv_size, bv_size, n_views=6):
        super(ViewTransformation, self).__init__()
        self.n_views = n_views
        self.hw_mat = []
        self.bv_size = bv_size
        fv_dim = fv_size[0] * fv_size[1]
        bv_dim = bv_size[0] * bv_size[1]
        for i in range(self.n_views):
            fc_transform = nn.Sequential(
                nn.Linear(fv_dim, bv_dim),
                nn.ReLU(),
                nn.Linear(bv_dim, bv_dim),
                nn.ReLU()
            )
            self.hw_mat.append(fc_transform)
        self.hw_mat = nn.ModuleList(self.hw_mat)

    def forward(self, feat):
        B, N, C, H, W = feat.shape
        feat = feat.view(B, N, C, H*W)
        outputs = []
        for i in range(N):
            output = self.hw_mat[i](feat[:, i]).view(
                B, C, self.bv_size[0], self.bv_size[1])
            outputs.append(output)
        outputs = torch.stack(outputs, 1)
        return outputs

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class CamEncode(nn.Module):
    
    def __init__(self, C):
        super(CamEncode, self).__init__()
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.up1 = Up(320+112, self.C)

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        return self.get_eff_depth(x)


@BACKBONES.register_module()
class HDMapNetBackbone(nn.Module):
    def __init__(self,
                 img_res,
                 out_channels,
                 canvas_size,
                 n_views,
                 up_sample_scale=2):
        super(HDMapNetBackbone, self).__init__()

        self.out_channels = out_channels
        self.downsample = 16
        self.canvas_size = canvas_size

        self.camencode = CamEncode(out_channels)
        self.fv_size = (img_res[0]//self.downsample, img_res[1]//self.downsample)

        canvas_h, canvas_w = canvas_size[1], canvas_size[0]
        self.bv_size = (canvas_h//5, canvas_w//5) # (40, 80)
        
        self.view_fusion = ViewTransformation(fv_size=self.fv_size, bv_size=self.bv_size, n_views=n_views)

        # Use the closest 3/4 part to do ipm
        res_x = self.bv_size[1] * 3 // 4
        res_y = self.bv_size[0] * 3 // 4
        ipm_xbound = [-res_x, res_x, 4 * res_x / canvas_size[0]] # [-60, 60, 0.6]
        ipm_ybound = [-res_y, res_y, 4 * res_y / canvas_size[1]] # [-30, 30, 0.6]
        self.ipm = IPM(ipm_xbound, ipm_ybound, extrinsic=True)
        self.up_sampler = nn.Upsample(
            scale_factor=up_sample_scale, mode='bilinear', align_corners=True)

    def get_Ks(self, intrins):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(
            1, 1, 4, 4).repeat(B, N, 1, 1)

        return Ks

    def get_cam_feats(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.out_channels, imH//self.downsample, imW//self.downsample)
        return x

    def forward(self, imgs, img_metas):
        x = self.get_cam_feats(imgs)
        x = self.view_fusion(x)

        device = x.device

        # prepare batched img_metas
        desired_keys = ['cam_intrinsics','cam_extrinsics']
        
        batched_img_metas = {k: [] for k in img_metas[0].keys() if k in desired_keys}
        
        for img_meta in img_metas:
            for k in desired_keys:
                batched_img_metas[k].append(img_meta[k])
        
        for k in batched_img_metas.keys():
            batched_img_metas[k] = torch.tensor(batched_img_metas[k]).to(device)
        
        RTs = batched_img_metas['cam_extrinsics'].float()
        Ks = self.get_Ks(batched_img_metas['cam_intrinsics'])

        # from camera system to BEV
        topdown = self.ipm(x, Ks, RTs)
        topdown = self.up_sampler(topdown)

        return topdown

