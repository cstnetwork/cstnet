import torch
import torch.nn as nn
import torch.nn.functional as F
from models import utils


class QuarticSSA(nn.Module):
    def __init__(self, n_center, n_near, n_stepk, n_primitive, channel_in, channel_out):
        """
        n_center: sampled points in this layer
        n_near: neighborhoods of center points
        n_stepk: each step neighbor in SurfaceKNN
        mlp: channels of MLP

        """
        super().__init__()
        self.n_center = n_center
        self.n_near = n_near
        self.n_stepk = n_stepk

        self.attention_points_mad = utils.PointAttention(channel_in, channel_in + 3, channel_out)
        self.attention_points_adj = utils.PointAttention(channel_in, channel_in + 4, channel_out)
        self.attention_points_pt = utils.PointAttention(channel_in, channel_in + n_primitive * 2, channel_out)
        self.attention_points_cst = utils.PointAttention(channel_in, channel_in + 3, channel_out)

        self.mlp_mad = utils.MLP(0, (channel_in + 3, channel_out, channel_out))
        self.mlp_adj = utils.MLP(0, (channel_in + 4, channel_out, channel_out))
        self.mlp_pt = utils.MLP(0, (channel_in + n_primitive * 2, channel_out, channel_out))
        self.mlp_cst = utils.MLP(0, (channel_in + 3, channel_out, channel_out))

    def forward(self, xyz, mad, adj, pt, mad_fea, adj_fea, pt_fea, cst_fea):
        """
        xyz: [bs, n_point, 3]
        mad: [bs, n_point, 3]
        adj: [bs, n_point, 2] one-hot
        pt: [bs, n_point, 4] one-hot

        mad_fea: [bs, n_point, n_channel]
        adj_fea: [bs, n_point, n_channel]
        pt_fea: [bs, n_point, n_channel]
        cst_fea: [bs, n_point, n_channel]

        """

        if self.n_center is None:  # group all point into a center

            g_mad = mad
            g_adj = adj.repeat(1, 1, 2)
            g_pt = pt.repeat(1, 1, 2)
            g_xyz = xyz

            g_mad_fea = mad_fea
            g_adj_fea = adj_fea
            g_pt_fea = pt_fea
            g_cst_fea = cst_fea

            g_mad_fea = torch.cat([g_mad_fea, g_mad], dim=-1).max(1)[0]
            g_adj_fea = torch.cat([g_adj_fea, g_adj], dim=-1).max(1)[0]
            g_pt_fea = torch.cat([g_pt_fea, g_pt], dim=-1).max(1)[0]
            g_cst_fea = torch.cat([g_cst_fea, g_xyz], dim=-1).max(1)[0]

            g_mad_fea = self.mlp_mad(g_mad_fea)
            g_adj_fea = self.mlp_adj(g_adj_fea)
            g_pt_fea = self.mlp_pt(g_pt_fea)
            g_cst_fea = self.mlp_cst(g_cst_fea)

            return g_mad_fea, g_adj_fea, g_pt_fea, g_cst_fea

        else:
            idx_surfknn_all = utils.surface_knn(xyz, self.n_near, self.n_stepk)  # [bs, n_point, n_near]
            fps_idx = utils.fps(xyz, self.n_center)  # [bs, n_center]

            center_xyz = utils.index_points(xyz, fps_idx)  # [bs, n_center, 3]
            center_mad = utils.index_points(mad, fps_idx)
            center_adj = utils.index_points(adj, fps_idx)
            center_pt = utils.index_points(pt, fps_idx)

            center_mad_fea = utils.index_points(mad_fea, fps_idx)  # [bs, n_center, f]
            center_adj_fea = utils.index_points(adj_fea, fps_idx)
            center_pt_fea = utils.index_points(pt_fea, fps_idx)
            center_cst_fea = utils.index_points(cst_fea, fps_idx)

            idx = utils.index_points(idx_surfknn_all, fps_idx)  # [bs, n_center, n_near]

            g_xyz = utils.index_points(xyz, idx)  # [bs, n_center, n_near, 3]
            g_mad = utils.index_points(mad, idx)
            g_adj = utils.index_points(adj, idx)
            g_pt = utils.index_points(pt, idx)

            g_mad_fea = utils.index_points(mad_fea, idx)  # [bs, n_center, n_near, 3]
            g_adj_fea = utils.index_points(adj_fea, idx)
            g_pt_fea = utils.index_points(pt_fea, idx)
            g_cst_fea = utils.index_points(cst_fea, idx)

            mad_relative = g_mad - center_mad.unsqueeze(2)  # [bs, n_center, n_near, 3]
            adj_cat = torch.cat([g_adj, center_adj.unsqueeze(2).repeat(1, 1, self.n_near, 1)], dim=-1)
            pt_cat = torch.cat([g_pt, center_pt.unsqueeze(2).repeat(1, 1, self.n_near, 1)], dim=-1)
            xyz_relative = g_xyz - center_xyz.unsqueeze(2)

            g_mad_fea = torch.cat([g_mad_fea, mad_relative], dim=-1)  # [bs, n_center, n_near, f]
            g_adj_fea = torch.cat([g_adj_fea, adj_cat], dim=-1)
            g_pt_fea = torch.cat([g_pt_fea, pt_cat], dim=-1)
            g_cst_fea = torch.cat([g_cst_fea, xyz_relative], dim=-1)

            g_mad_fea = self.attention_points_mad(center_mad_fea, g_mad_fea)
            g_adj_fea = self.attention_points_adj(center_adj_fea, g_adj_fea)
            g_pt_fea = self.attention_points_pt(center_pt_fea, g_pt_fea)
            g_cst_fea = self.attention_points_cst(center_cst_fea, g_cst_fea)

            return center_xyz, center_mad, center_adj, center_pt, g_mad_fea, g_adj_fea, g_pt_fea, g_cst_fea


class CMLP(nn.Module):
    def __init__(self, n_primitive=4, mlp_ex0=(16, 32)):
        """
        n_primitive: classes number of primitives
        mlp_ex0: mlp channels except layer 0
        """
        super().__init__()
        self.cst_mlp = utils.MLP(1, (3 + 3 + 2 + n_primitive,) + mlp_ex0)

    def forward(self, xyz, mad, adj, pt):
        """
        xyz: [bs, n_point, 3]
        mad: [bs, n_point, 3]
        adj: [bs, n_point, 2] one-hot
        pt: [bs, n_point, 4] one-hot
        fea: [bs, n_point, n_channel] one-hot
        """
        xyz_cst = torch.cat([xyz, mad, adj, pt], dim=2).permute(0, 2, 1)
        cst_fea = self.cst_mlp(xyz_cst).permute(0, 2, 1)

        return cst_fea


class TripleMLP(nn.Module):
    def __init__(self, n_primitive=4, mlp_ex0=(16, 32)):
        """
        n_primitive: classes number of primitives
        mlp_ex0: mlp channels except layer 0
        """
        super().__init__()

        self.mlp_mad = utils.MLP(1, (3 + 3,) + mlp_ex0)
        self.mlp_adj = utils.MLP(1, (3 + 2,) + mlp_ex0)
        self.mlp_pt = utils.MLP(1, (3 + n_primitive,) + mlp_ex0)

    def forward(self, xyz, mad, adj, pt):
        """
        xyz: [bs, n_point, 3]
        mad: [bs, n_point, 3]
        adj: [bs, n_point, 2] one-hot
        pt: [bs, n_point, 4] one-hot
        """
        xyz_mad = torch.cat([xyz, mad], dim=2).permute(0, 2, 1)
        xyz_adj = torch.cat([xyz, adj], dim=2).permute(0, 2, 1)
        xyz_pt = torch.cat([xyz, pt], dim=2).permute(0, 2, 1)

        mad_fea = self.mlp_mad(xyz_mad).permute(0, 2, 1)
        adj_fea = self.mlp_adj(xyz_adj).permute(0, 2, 1)
        pt_fea = self.mlp_pt(xyz_pt).permute(0, 2, 1)

        return mad_fea, adj_fea, pt_fea


class CstNetS2(nn.Module):
    def __init__(self, n_classes, n_primitive):
        super().__init__()
        print('creating CstNet Stage2 network ...')

        channel_l0 = 32
        channel_l1 = 64
        channel_l2 = 128
        channel_l3 = 256
        channel_l4 = 512
        channel_l5 = 1024

        self.tri_mlp = TripleMLP(n_primitive, (16, channel_l0))
        self.c_mlp = CMLP(n_primitive, (16, channel_l0))

        self.quartic_ssa1 = QuarticSSA(1024, 32, 5, n_primitive, channel_l0, channel_l1)
        self.fea_attn1 = utils.FeaAttention(channel_l1, channel_l1)

        self.quartic_ssa2 = QuarticSSA(512, 32, 5, n_primitive, channel_l1, channel_l2)
        self.fea_attn2 = utils.FeaAttention(channel_l2, channel_l2)

        self.quartic_ssa3 = QuarticSSA(128, 64, 10, n_primitive, channel_l2, channel_l3)
        self.fea_attn3 = utils.FeaAttention(channel_l3, channel_l3)

        self.quartic_ssa4 = QuarticSSA(64, 64, 10, n_primitive, channel_l3, channel_l4)
        self.fea_attn4 = utils.FeaAttention(channel_l4, channel_l4)

        self.quartic_ssa5 = QuarticSSA(None, None, None, n_primitive, channel_l4, channel_l5)
        self.fea_attn5 = utils.FeaAttention(channel_l5, channel_l5)

        self.linear = utils.MLP(0, (channel_l5, channel_l4, channel_l3, channel_l2, n_classes))

    def forward(self, xyz, mad, adj, pt):
        """
        xyz: [bs, n_point, 3]
        mad: [bs, n_point, 3]
        adj: [bs, n_point, 2]
        pt: [bs, n_point, 4]

        """
        mad_fea, adj_fea, pt_fea = self.tri_mlp(xyz, mad, adj, pt)
        cst_fea = self.c_mlp(xyz, mad, adj, pt)

        xyz_l1, mad_l1, adj_l1, pt_l1, mad_fea_l1, adj_fea_l1, pt_fea_l1, cst_fea_l1 = self.quartic_ssa1(xyz, mad, adj, pt, mad_fea, adj_fea, pt_fea, cst_fea)
        cst_fea_l1 = self.fea_attn1(mad_fea_l1, adj_fea_l1, pt_fea_l1, cst_fea_l1)

        xyz_l2, mad_l2, adj_l2, pt_l2, mad_fea_l2, adj_fea_l2, pt_fea_l2, cst_fea_l2 = self.quartic_ssa2(xyz_l1, mad_l1, adj_l1, pt_l1, mad_fea_l1, adj_fea_l1, pt_fea_l1, cst_fea_l1)
        cst_fea_l2 = self.fea_attn2(mad_fea_l2, adj_fea_l2, pt_fea_l2, cst_fea_l2)

        xyz_l3, mad_l3, adj_l3, pt_l3, mad_fea_l3, adj_fea_l3, pt_fea_l3, cst_fea_l3 = self.quartic_ssa3(xyz_l2, mad_l2, adj_l2, pt_l2, mad_fea_l2, adj_fea_l2, pt_fea_l2, cst_fea_l2)
        cst_fea_l3 = self.fea_attn3(mad_fea_l3, adj_fea_l3, pt_fea_l3, cst_fea_l3)

        xyz_l4, mad_l4, adj_l4, pt_l4, mad_fea_l4, adj_fea_l4, pt_fea_l4, cst_fea_l4 = self.quartic_ssa4(xyz_l3, mad_l3, adj_l3, pt_l3, mad_fea_l3, adj_fea_l3, pt_fea_l3, cst_fea_l3)
        cst_fea_l4 = self.fea_attn4(mad_fea_l4, adj_fea_l4, pt_fea_l4, cst_fea_l4)

        mad_fea_l5, adj_fea_l5, pt_fea_l5, cst_fea_l5 = self.quartic_ssa5(xyz_l4, mad_l4, adj_l4, pt_l4, mad_fea_l4, adj_fea_l4, pt_fea_l4, cst_fea_l4)
        cst_fea_l5 = self.fea_attn5(mad_fea_l5.unsqueeze(1), adj_fea_l5.unsqueeze(1), pt_fea_l5.unsqueeze(1), cst_fea_l5.unsqueeze(1)).squeeze()

        cls = self.linear(cst_fea_l5)
        cls = F.log_softmax(cls, -1)
        return cls


if __name__ == '__main__':
    xyz_tensor = torch.rand(2, 2500, 3).cuda()
    mad_tensor = torch.rand(2, 2500, 3).cuda()
    adj_tensor = torch.rand(2, 2500, 2).cuda()
    pt_tensor = torch.rand(2, 2500, 4).cuda()

    anet = CstNetS2(10, 4).cuda()

    pred = anet(xyz_tensor, mad_tensor, adj_tensor, pt_tensor)

    print(pred.shape)
    print(pred)

