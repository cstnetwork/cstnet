import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sortedcontainers import SortedDict
import torch.nn.functional as F
import time
from scipy.interpolate import griddata
import math

import pymeshlab
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_rectangular_prism(ax, origin, size):
    x = [origin[0], origin[0] + size[0]]
    y = [origin[1], origin[1] + size[1]]
    z = [origin[2], origin[2] + size[2]]

    vertices = [[x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[0], y[1], z[0]],
                [x[0], y[0], z[1]], [x[1], y[0], z[1]], [x[1], y[1], z[1]], [x[0], y[1], z[1]]]

    faces = [[vertices[j] for j in [0, 1, 5, 4]],
             [vertices[j] for j in [7, 6, 2, 3]],
             [vertices[j] for j in [0, 3, 7, 4]],
             [vertices[j] for j in [1, 2, 6, 5]],
             [vertices[j] for j in [0, 1, 2, 3]],
             [vertices[j] for j in [4, 5, 6, 7]]]

    poly3d = Poly3DCollection(faces, alpha=0.1, edgecolor='k', facecolors=[1,1,1])

    ax.add_collection3d(poly3d)


class full_connected_conv3d(nn.Module):
    def __init__(self, channels: list, bias: bool = True, drop_rate: float = 0.4):
        super().__init__()

        self.linear_layers = nn.ModuleList()
        self.batch_normals = nn.ModuleList()
        self.drop_outs = nn.ModuleList()
        self.n_layers = len(channels)

        for i in range(self.n_layers - 2):
            self.linear_layers.append(nn.Conv3d(channels[i], channels[i + 1], 1, bias=bias))
            self.batch_normals.append(nn.BatchNorm3d(channels[i + 1]))
            self.drop_outs.append(nn.Dropout3d(drop_rate))

        self.outlayer = nn.Conv3d(channels[-2], channels[-1], 1, bias=bias)

    def forward(self, embeddings):
        fea = embeddings
        for i in range(self.n_layers - 2):
            fc = self.linear_layers[i]
            bn = self.batch_normals[i]
            drop = self.drop_outs[i]

            fea = drop(F.relu(bn(fc(fea))))

        fea = self.outlayer(fea)

        return fea


class full_connected_conv2d(nn.Module):
    def __init__(self, channels: list, bias: bool = True, drop_rate: float = 0.4):
        super().__init__()

        self.linear_layers = nn.ModuleList()
        self.batch_normals = nn.ModuleList()
        self.drop_outs = nn.ModuleList()
        self.n_layers = len(channels)

        for i in range(self.n_layers - 2):
            self.linear_layers.append(nn.Conv2d(channels[i], channels[i + 1], 1, bias=bias))
            self.batch_normals.append(nn.BatchNorm2d(channels[i + 1]))
            self.drop_outs.append(nn.Dropout2d(drop_rate))

        self.outlayer = nn.Conv2d(channels[-2], channels[-1], 1, bias=bias)

    def forward(self, embeddings):
        fea = embeddings
        for i in range(self.n_layers - 2):
            fc = self.linear_layers[i]
            bn = self.batch_normals[i]
            drop = self.drop_outs[i]

            fea = drop(F.relu(bn(fc(fea))))

        fea = self.outlayer(fea)

        return fea


class full_connected_conv1d(nn.Module):
    def __init__(self, channels: list, bias: bool = True, drop_rate: float = 0.4):
        super().__init__()

        self.linear_layers = nn.ModuleList()
        self.batch_normals = nn.ModuleList()
        self.drop_outs = nn.ModuleList()
        self.n_layers = len(channels)

        for i in range(self.n_layers - 2):
            self.linear_layers.append(nn.Conv1d(channels[i], channels[i + 1], 1, bias=bias))
            self.batch_normals.append(nn.BatchNorm1d(channels[i + 1]))
            self.drop_outs.append(nn.Dropout1d(drop_rate))

        self.outlayer = nn.Conv1d(channels[-2], channels[-1], 1, bias=bias)

    def forward(self, embeddings):
        fea = embeddings
        for i in range(self.n_layers - 2):
            fc = self.linear_layers[i]
            bn = self.batch_normals[i]
            drop = self.drop_outs[i]

            fea = drop(F.relu(bn(fc(fea))))

        fea = self.outlayer(fea)

        return fea


class full_connected(nn.Module):
    def __init__(self, channels: list, bias: bool = True, drop_rate: float = 0.4):
        super().__init__()

        self.linear_layers = nn.ModuleList()
        self.batch_normals = nn.ModuleList()
        self.drop_outs = nn.ModuleList()
        self.n_layers = len(channels)

        for i in range(self.n_layers - 2):
            self.linear_layers.append(nn.Linear(channels[i], channels[i + 1], bias=bias))
            self.batch_normals.append(nn.BatchNorm1d(channels[i + 1]))
            self.drop_outs.append(nn.Dropout(drop_rate))

        self.outlayer = nn.Linear(channels[-2], channels[-1], bias=bias)

    def forward(self, embeddings):
        fea = embeddings
        for i in range(self.n_layers - 2):
            fc = self.linear_layers[i]
            bn = self.batch_normals[i]
            drop = self.drop_outs[i]

            fea = drop(F.relu(bn(fc(fea))))

        fea = self.outlayer(fea)

        return fea


class SA_Attention(nn.Module):
    def __init__(self, n_center, n_near, in_channel, mlp):
        super().__init__()

        self.n_center = n_center
        self.n_near = n_near
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)

        if points is not None:
            points = points.permute(0, 2, 1)

        new_xyz, new_points = SA_SampleAndGroup(self.n_center, self.n_near, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1).to(torch.float)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class SetAbstraction(nn.Module):
    def __init__(self, n_center, n_near, in_channel, mlp):
        super().__init__()

        self.n_center = n_center
        self.n_near = n_near
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)

        if points is not None:
            points = points.permute(0, 2, 1)

        new_xyz, new_points = SA_SampleAndGroup(self.n_center, self.n_near, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1).to(torch.float)  # [B, C+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class SA_Scratch(nn.Module):
    def __init__(self, n_center, n_near, in_channel, mlp):
        super().__init__()

        self.n_center = n_center
        self.n_near = n_near
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, fea):
        xyz = xyz.permute(0, 2, 1)

        if fea is not None:
            fea = fea.permute(0, 2, 1)

        new_xyz, new_fea = SA_SampleAndGroup(self.n_center, self.n_near, xyz, fea)

        new_fea = new_fea.permute(0, 3, 2, 1).to(torch.float)  # [B, C+D, n_near, n_center]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_fea = F.relu(bn(conv(new_fea)))

        new_fea = torch.max(new_fea, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_fea


class SA_Attention_test3(nn.Module):
    def __init__(self, n_center, n_near, dim_in, mlp, dim_qkv):
        super().__init__()

        self.n_center = n_center
        self.n_near = n_near
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = dim_in

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.matq = nn.Conv2d(last_channel, dim_qkv, 1, bias=False)
        self.matk = nn.Conv2d(last_channel, dim_qkv, 1, bias=False)
        self.matv = nn.Conv2d(last_channel, dim_qkv, 1, bias=False)

        self.pos_mlp = full_connected_conv2d([last_channel, last_channel, dim_qkv])
        self.weight_mlp = full_connected_conv2d([dim_qkv, dim_qkv, dim_qkv])

        self.fea_final = full_connected_conv1d([dim_qkv, dim_qkv, dim_qkv])

    def forward(self, xyz, fea):
        new_xyz, new_fea, new_fea_center = SA_SampleAndGroup(self.n_center, self.n_near, xyz, fea, is_backecenter=True)
        #        new_fea -> [bs, n_center, n_near, 3+emb_in]
        # new_fea_center -> [bs, n_center, 1     , 3+emb_in]

        new_fea = new_fea.permute(0, 3, 2, 1).to(torch.float)  # -> [B, C+D, n_near, n_center]
        new_fea_center = new_fea_center.permute(0, 3, 2, 1).to(torch.float)  # -> [B, C+D, 1, n_center]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_fea = F.relu(bn(conv(new_fea)))
            new_fea_center = F.relu(bn(conv(new_fea_center)))

        center_fea_orig = new_fea_center
        # ->[bs, emb_in, 1, n_center]

        grouped_fea = new_fea
        # ->[bs, emb_in, n_near, n_center]

        center_q = self.matq(center_fea_orig)
        center_k = self.matk(center_fea_orig)
        # ->[bs, dim_qkv, 1, n_center]

        near_k = self.matk(grouped_fea)
        # ->[bs, dim_qkv, n_near, n_center]

        center_v = self.matv(center_fea_orig)
        # ->[bs, dim_qkv, 1, n_center]

        near_v = self.matv(grouped_fea)
        # ->[bs, dim_qkv, n_near, n_center]

        cen_near_k = torch.cat([center_k, near_k], dim=2)
        # ->[bs, dim_qkv, 1 + n_near, n_center]

        cen_near_v = torch.cat([center_v, near_v], dim=2)
        # ->[bs, dim_qkv, 1 + n_near, n_center]

        emb_pos = self.pos_mlp(center_fea_orig - torch.cat([center_fea_orig, grouped_fea], dim=2))
        # ->[bs, dim_qkv, 1 + n_near, n_center]

        # MLP(q - k + delta)
        emb_weight = self.weight_mlp(center_q - cen_near_k + emb_pos)
        # ->[bs, dim_qkv, 1 + n_near, n_center]

        emb_weight = F.softmax(emb_weight, dim=2)
        # ->[bs, dim_qkv, 1 + n_near, n_center]

        center_fea = emb_weight * (cen_near_v + emb_pos)
        # ->[bs, dim_qkv, 1 + n_near, n_center]

        center_fea = torch.sum(center_fea, dim=2)
        # ->[bs, dim_qkv, n_center]

        center_fea = self.fea_final(center_fea)
        # ->[bs, dim_out, n_center]

        center_fea = center_fea.permute(0, 2, 1)
        # ->[bs, n_center, dim_out]

        return center_fea, new_fea


class SA_Attention_test2(nn.Module):
    def __init__(self, n_center, n_near, dim_in, dim_qk, dim_v, mlp: list):
        super().__init__()

        self.n_center = n_center
        self.n_near = n_near

        self.matq = nn.Conv2d(dim_in, dim_qk, 1, bias=False)
        self.matk = nn.Conv2d(dim_in, dim_qk, 1, bias=False)
        self.matv = nn.Conv2d(dim_in, dim_v, 1, bias=False)

        self.fea_final = full_connected_conv1d(mlp)

    def forward(self, xyz, fea):
        idx_surfknn_all = surface_knn(xyz, self.n_near, 10)

        if self.n_center is None:
            center_xyz = xyz
            center_fea_orig = fea
            idx = idx_surfknn_all
            grouped_xyz = index_points(xyz, idx_surfknn_all)

        else:
            fps_idx = farthest_point_sample(xyz, self.n_center)
            center_xyz = index_points(xyz, fps_idx)
            center_fea_orig = index_points(fea, fps_idx)
            idx = index_points(idx_surfknn_all, fps_idx)
            grouped_xyz = index_points(xyz, idx)  # [B, n_center, n_near, 3]

        if fea is not None:
            grouped_fea = index_points(fea, idx)
        else:
            grouped_fea = grouped_xyz  # [B, n_center, n_near, dim_in]

        center_fea_orig_forcat = center_fea_orig
        center_fea_orig = center_fea_orig.unsqueeze(2)
        # <-[bs, n_center, dim_in]
        # ->[bs, n_center, 1, dim_in]

        center_fea_orig = center_fea_orig.permute(0, 3, 2, 1)
        # ->[bs, dim_in, 1, n_center]

        grouped_fea = grouped_fea.permute(0, 3, 2, 1)
        # ->[bs, dim_in, n_near, n_center]

        center_q = self.matq(center_fea_orig)
        center_k = self.matk(center_fea_orig)
        # ->[bs, dim_qk, 1, n_center]

        near_k = self.matk(grouped_fea)
        # ->[bs, dim_qk, n_near, n_center]

        center_v = self.matv(center_fea_orig)
        # ->[bs, dim_v, 1, n_center]

        near_v = self.matv(grouped_fea)
        # ->[bs, dim_v, n_near, n_center]

        cen_near_k = torch.cat([center_k, near_k], dim=2)
        # ->[bs, dim_qk, 1 + n_near, n_center]

        cen_near_v = torch.cat([center_v, near_v], dim=2)
        # ->[bs, dim_v, 1 + n_near, n_center]

        # q * k
        q_dot_k = torch.sum(center_q * cen_near_k, dim=1, keepdim=True)
        # ->[bs, 1, 1 + n_near, n_center]
        q_dot_k = F.softmax(q_dot_k, dim=2)

        # (q dot k) * v
        weighted_v = q_dot_k * cen_near_v
        # ->[bs, dim_v, 1 + n_near, n_center]

        # 求属性加权和
        center_fea = torch.sum(weighted_v, dim=2)
        # ->[bs, dim_v, n_center]

        center_fea = self.fea_final(center_fea)
        # ->[bs, dim_out, n_center]

        center_fea = center_fea.permute(0, 2, 1)
        # ->[bs, n_center, dim_out]

        center_fea = torch.cat([center_fea_orig_forcat, center_fea], dim=-1)
        # ->[bs, n_center, emb_in + dim_out]

        return center_xyz, center_fea


class SA_Attention_test1(nn.Module):
    def __init__(self, n_center, n_near, dim_in, dim_qkv, mlp: list):
        super().__init__()

        self.n_center = n_center
        self.n_near = n_near

        self.matq = nn.Conv2d(dim_in, dim_qkv, 1, bias=False)
        self.matk = nn.Conv2d(dim_in, dim_qkv, 1, bias=False)
        self.matv = nn.Conv2d(dim_in, dim_qkv, 1, bias=False)

        self.pos_mlp = full_connected_conv2d([dim_in, dim_in, dim_qkv])
        self.weight_mlp = full_connected_conv2d([dim_qkv, dim_qkv, dim_qkv])

        self.fea_final = full_connected_conv1d(mlp)

    def forward(self, xyz, fea):
        idx_surfknn_all = surface_knn(xyz, self.n_near, 10)

        if self.n_center is None:
            center_xyz = xyz
            center_fea_orig = fea
            idx = idx_surfknn_all
            grouped_xyz = index_points(xyz, idx_surfknn_all)

        else:
            fps_idx = farthest_point_sample(xyz, self.n_center)
            center_xyz = index_points(xyz, fps_idx)
            center_fea_orig = index_points(fea, fps_idx)
            idx = index_points(idx_surfknn_all, fps_idx)
            grouped_xyz = index_points(xyz, idx)  # [B, n_center, n_near, 3]

        if fea is not None:
            grouped_fea = index_points(fea, idx)
        else:
            grouped_fea = grouped_xyz  # [B, n_center, n_near, emb_in]

        center_fea_orig_forcat = center_fea_orig
        center_fea_orig = center_fea_orig.unsqueeze(2)
        # <-[bs, n_center, emb_in]
        # ->[bs, n_center, 1, emb_in]

        center_fea_orig = center_fea_orig.permute(0, 3, 2, 1)
        # ->[bs, emb_in, 1, n_center]

        grouped_fea = grouped_fea.permute(0, 3, 2, 1)
        # ->[bs, emb_in, n_near, n_center]

        center_q = self.matq(center_fea_orig)
        center_k = self.matk(center_fea_orig)
        # ->[bs, dim_qkv, 1, n_center]

        near_k = self.matk(grouped_fea)
        # ->[bs, dim_qkv, n_near, n_center]

        center_v = self.matv(center_fea_orig)
        # ->[bs, dim_qkv, 1, n_center]

        near_v = self.matv(grouped_fea)
        # ->[bs, dim_qkv, n_near, n_center]

        cen_near_k = torch.cat([center_k, near_k], dim=2)
        # ->[bs, dim_qkv, 1 + n_near, n_center]

        cen_near_v = torch.cat([center_v, near_v], dim=2)
        # ->[bs, dim_qkv, 1 + n_near, n_center]

        emb_pos = self.pos_mlp(center_fea_orig - torch.cat([center_fea_orig, grouped_fea], dim=2))
        # ->[bs, dim_qkv, 1 + n_near, n_center]

        # MLP(q - k + delta)
        emb_weight = self.weight_mlp(center_q - cen_near_k + emb_pos)
        # ->[bs, dim_qkv, 1 + n_near, n_center]

        emb_weight = F.softmax(emb_weight, dim=2)
        # ->[bs, dim_qkv, 1 + n_near, n_center]

        center_fea = emb_weight * (cen_near_v + emb_pos)
        # ->[bs, dim_qkv, 1 + n_near, n_center]

        center_fea = torch.sum(center_fea, dim=2)
        # ->[bs, dim_qkv, n_center]

        center_fea = self.fea_final(center_fea)
        # ->[bs, dim_out, n_center]

        center_fea = center_fea.permute(0, 2, 1)
        # ->[bs, n_center, dim_out]

        center_fea = torch.cat([center_fea_orig_forcat, center_fea], dim=-1)
        # ->[bs, n_center, emb_in + dim_out]

        return center_xyz, center_fea


def index_points(points, idx, is_label: bool = False):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    if is_label:
        new_points = points[batch_indices, idx]
    else:
        new_points = points[batch_indices, idx, :]

    return new_points


class FeaPropagate(nn.Module):
    def __init__(self, in_channel, mlp):  # fp1: in_channel=150, mlp=[128, 128]
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # ->[B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            # skip link concatenation
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def SA_SampleAndGroup(n_center, n_near, xyz, fea, is_backecenter=False):
    idx_surfknn_all = surface_knn(xyz, n_near, 10)

    if n_center is None:
        new_xyz = xyz
        idx = idx_surfknn_all
        grouped_xyz = index_points(xyz, idx_surfknn_all)

    else:
        fps_idx = farthest_point_sample(xyz, n_center)
        new_xyz = index_points(xyz, fps_idx)
        idx = index_points(idx_surfknn_all, fps_idx)
        grouped_xyz = index_points(xyz, idx)

    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

    if fea is not None:
        grouped_fea = index_points(fea, idx)
        new_fea = torch.cat([grouped_xyz_norm, grouped_fea], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_fea = grouped_xyz_norm

    if is_backecenter:
        if n_center is None:
            new_fea_center = fea
        else:
            new_fea_center = index_points(fea, fps_idx)

        grouped_xyz_norm_center = torch.zeros_like(new_xyz)
        # ->[bs, n_center, 3]

        new_fea_center = torch.cat([grouped_xyz_norm_center, new_fea_center], dim=-1).unsqueeze(2)
        # ->[bs, n_center, 1, 3+emb_in]

        return new_xyz, new_fea, new_fea_center
    else:
        return new_xyz, new_fea


def farthest_point_sample(xyz, n_samples):
    device = xyz.device

    # xyz: [24, 1024, 3], B: batch_size, N: number of points, C: channels
    B, N, C = xyz.shape

    centroids = torch.zeros(B, n_samples, dtype=torch.long).to(device)

    distance = torch.ones(B, N).to(device) * 1e10

    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(n_samples):
        centroids[:, i] = farthest

        # print('batch_indices', batch_indices.shape)
        # print('farthest', farthest.shape)
        # print('xyz', xyz[batch_indices, farthest, :].shape)
        # exit()

        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)

        # print('xyz', xyz.shape)
        # print('centroid', centroid.shape)
        # exit()

        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask].float()
        farthest = torch.max(distance, -1)[1]
    return centroids


def get_neighbor_index(vertices: "(bs, vertice_num, 3)",  neighbor_num: int, is_backdis: bool = False):
    bs, v, _ = vertices.size()
    inner = torch.bmm(vertices, vertices.transpose(1, 2)) #(bs, v, v)
    quadratic = torch.sum(vertices**2, dim= 2) #(bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    # print('distance.shape: ', distance.shape)

    neighbor_index = torch.topk(distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    if is_backdis:
        return neighbor_index, distance
    else:
        return neighbor_index


def indexes_val(vals, inds):
    bs, n_item, n_vals = inds.size()

    sequence = torch.arange(bs)
    sequence_expanded = sequence.unsqueeze(1)
    sequence_3d = sequence_expanded.tile((1, n_item))
    sequence_4d = sequence_3d.unsqueeze(-1)
    batch_indices = sequence_4d.repeat(1, 1, n_vals)

    view_shape = [n_item, n_vals]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = [bs, n_item, n_vals]
    repeat_shape[1] = 1
    channel_indices = torch.arange(n_item, dtype=torch.long).view(view_shape).repeat(repeat_shape)

    return vals[batch_indices, channel_indices, inds]


def surface_knn(points_all: "(bs, n_pnts, 3)", k_near: int = 100, n_stepk = 10):
    ind_neighbor_all, all_dist = get_neighbor_index(points_all, n_stepk, True)

    neighbor_index_max = torch.max(all_dist, dim=-1, keepdim=True)[1]

    new_neighinds = ind_neighbor_all.clone()

    num_ita = 0
    while True:
        n_current_neighbors = new_neighinds.size()[-1]
        indexed_all = []
        for j in range(n_current_neighbors):
            indexed_all.append(index_points(ind_neighbor_all, new_neighinds[:, :, j]))
        new_neighinds = torch.cat(indexed_all, dim=-1)

        new_neighinds = torch.sort(new_neighinds, dim=-1)[0]

        duplicates = torch.zeros_like(new_neighinds)
        duplicates[:, :, 1:] = new_neighinds[:, :, 1:] == new_neighinds[:, :, :-1]

        neighbor_index_max2 = neighbor_index_max.repeat(1, 1, new_neighinds.shape[-1])
        new_neighinds[duplicates.bool()] = neighbor_index_max2[duplicates.bool()]

        dist_neighinds = indexes_val(all_dist, new_neighinds)

        sort_dist = torch.sort(dist_neighinds, dim=-1)[0]  # -> [bs, n_point, n_near]

        sort_dist_maxind = torch.max(sort_dist, dim=-1)[1]  # -> [bs, n_point]
        valid_nnear = torch.min(sort_dist_maxind).item() + 1

        is_end_loop = False
        if valid_nnear >= k_near + 1:
            valid_nnear = k_near + 1
            is_end_loop = True

        sub_neighbor_index = torch.topk(dist_neighinds, k=valid_nnear, dim=-1, largest=False)[1]  # [0] val, [1] index

        new_neighinds = indexes_val(new_neighinds, sub_neighbor_index)

        new_neighinds = new_neighinds[:, :, 1:]

        if is_end_loop:
            break

        num_ita += 1
        if num_ita > 20:
            print('max surface knn iteration count, return knn')
            return ind_neighbor_all

    return new_neighinds


if __name__ == '__main__':

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # origin = (0, 0, 0)
    # size = (3, 2, 1)
    #
    # plot_rectangular_prism(ax, origin, size)
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # ax.set_xlim([0, 5])
    # ax.set_ylim([0, 5])
    # ax.set_zlim([0, 5])
    #
    # plt.show()

    # vis_stl(r'C:\Users\ChengXi\Desktop\hardreads\cuboid.stl')

    # test()

    # surf_knn_pral()
    # show_surfknn_paper1()
    # show_surfknn_paper2()
    # show_surfknn_paper3()
    # show_surfknn_paper4()

    # teat_star()
    # test_where()
    # test_surfknn_testv2()
    # test_batch_indexes()
    # patch_interpolate()
    # test_unique()
    # test_knn2()

    # show_different_weight_paper()
    pass