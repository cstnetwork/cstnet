import torch.nn as nn
import torch
import torch.nn.functional as F

from models import utils


class SurfknnPntattn(nn.Module):
    def __init__(self, n_center, n_near, n_stepk, channel_in, channel_out):
        super().__init__()

        self.n_center = n_center
        self.n_near = n_near
        self.n_stepk = n_stepk

        self.attention_points = utils.PointAttention(channel_in, channel_in + 3, channel_out)

    def forward(self, xyz, fea):
        """
        xyz: [bs, 3, n_point]
        fea: [bs, f, n_point]
        """

        xyz = xyz.permute(0, 2, 1)
        fea = fea.permute(0, 2, 1)

        idx_surfknn_all = utils.surface_knn(xyz, self.n_near, self.n_stepk)  # [bs, n_point, n_near]
        fps_idx = utils.fps(xyz, self.n_center)  # [bs, n_center]
        idx = utils.index_points(idx_surfknn_all, fps_idx)  # [bs, n_center, n_near]

        center_xyz = utils.index_points(xyz, fps_idx)  # [bs, n_center, 3]
        g_xyz = utils.index_points(xyz, idx)  # [bs, n_center, n_near, 3]
        xyz_relative = g_xyz - center_xyz.unsqueeze(2)

        center_fea = utils.index_points(fea, fps_idx)
        g_fea = utils.index_points(fea, idx)
        g_fea = torch.cat([g_fea, xyz_relative], dim=-1)

        new_xyz = center_xyz.permute(0, 2, 1)
        new_fea = self.attention_points(center_fea, g_fea).permute(0, 2, 1)

        return new_xyz, new_fea


class UpSample(nn.Module):
    def __init__(self, mlp):  # fp1: in_channel=150, mlp=[128, 128]
        super().__init__()
        self.mlp = utils.MLP(1, mlp)

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        bs, n_point, channel = xyz1.shape

        dists = utils.square_distance(xyz1, xyz2)

        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)

        weight = dist_recip / norm  # ->[B, N, 3]
        interpolated_points = torch.sum(utils.index_points(points2, idx) * weight.view(bs, n_point, 3, 1), dim=2)

        # skip link concatenation
        points1 = points1.permute(0, 2, 1)
        new_points = torch.cat([points1, interpolated_points], dim=-1)

        new_points = new_points.permute(0, 2, 1)
        new_points = self.mlp(new_points)

        return new_points


class CstPnt(nn.Module):
    def __init__(self, n_points_all, n_primitive, n_embout=256, n_stepk=10, drate=0.9):
        super().__init__()

        self.sa1 = SurfknnPntattn(int(n_points_all * drate), 50, n_stepk, 3, 128)
        self.sa2 = SurfknnPntattn(int(n_points_all * drate ** 2), 100, n_stepk, 128, 256)

        self.fp2 = UpSample((256 + 128, 256, 256, 128))  # in_channel = points2_channel + points1_channel
        self.fp1 = UpSample((128 + 6, 128, 128, 128))

        self.mlp_fea = utils.MLP(1, (128, int((128 * n_embout) ** 0.5), n_embout))

        self.mlp_mad = utils.MLP(1, (n_embout, 256, 128, 32, 3))
        self.mlp_adj = utils.MLP(1, (n_embout, 256, 128, 32, 2))
        self.mlp_pt = utils.MLP(1, (n_embout, 256, 128, 64, n_primitive))

    def forward(self, xyz):
        """
        xyz: [bs, n_point, 3]
        """
        xyz = xyz.transpose(1, -1)

        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(xyz, xyz)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, torch.cat([xyz, xyz], 1), l1_points)

        # FC layers
        feat = self.mlp_fea(l0_points)
        ex_features = feat.permute(0, 2, 1)  # [bs, n_points_all, self.n_embout]

        ex_features = ex_features.transpose(-1, -2)
        # [bs, self.n_embout, n_points_all]

        mad = self.mlp_mad(ex_features).transpose(-1, -2)
        # [bs, n_points_all, 3]

        adj = self.mlp_adj(ex_features).transpose(-1, -2)
        # [bs, n_points_all, 2]

        pt = self.mlp_pt(ex_features).transpose(-1, -2)
        # [bs, n_points_all, 7]

        adj_log = F.log_softmax(adj, dim=-1)
        pt_log = F.log_softmax(pt, dim=-1)

        return mad, adj_log, pt_log


if __name__ == '__main__':
    test_tensor = torch.rand(2, 2500, 3).cuda()
    anet = CstPnt(2500, 4).cuda()

    _mad, _adj_log, _pt_log = anet(test_tensor)
    print(_mad.size(), _adj_log.size(), _pt_log.size())







