import torch.nn as nn
import torch
import torch.nn.functional as F

from models import utils


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


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


def knn(vertices: "(bs, vertice_num, 3)",  neighbor_num: int, is_backdis: bool = False):
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


def SA_SampleAndGroup(n_center, n_near, xyz, fea, is_backecenter=False):
    idx_surfknn_all = knn(xyz, n_near)

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


class SetAbstraction(nn.Module):
    def __init__(self, n_center, n_near, in_channel, mlp):
        super().__init__()

        self.n_center = n_center
        self.n_near = n_near
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:  # mlp：数组
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


class CstPnt(nn.Module):
    def __init__(self, n_points_all, n_primitive, n_embout=256, n_neighbor=100, n_stepk=10):
        super().__init__()

        self.n_neighbor = n_neighbor
        self.n_stepk = n_stepk

        rate_downsample = 0.9
        self.sa1 = SetAbstraction(n_center=int(n_points_all * rate_downsample), n_near=50, in_channel=3+3, mlp=[64, 64, 128])
        self.sa2 = SetAbstraction(n_center=int(n_points_all * rate_downsample ** 2), n_near=100, in_channel=128+3, mlp=[128, 128, 256])

        self.fp2 = FeaPropagate(in_channel=256+128, mlp=[256, 256, 128])  # in_chanell = points2_chanell + points1_channel
        self.fp1 = FeaPropagate(in_channel=128+6, mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, n_embout, 1)

        self.eula_angle = utils.MLP(1, (n_embout, 256, 128, 32, 3))

        self.edge_nearby = utils.MLP(1, (n_embout, 256, 128, 32, 2))

        self.meta_type = utils.MLP(1, (n_embout, 256, 128, 64, n_primitive))

    def forward(self, xyz):
        xyz = xyz.transpose(1, -1)

        # Set Abstraction layers
        l0_points = xyz
        l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        # Feature Propagation layers
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)

        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.drop1(feat)
        feat = self.conv2(feat)
        ex_features = feat.permute(0, 2, 1)
        # [bs, n_points_all, self.n_embout]

        ex_features = ex_features.transpose(-1, -2)
        # [bs, self.n_embout, n_points_all]

        eula_angle = self.eula_angle(ex_features).transpose(-1, -2)
        # [bs, n_points_all, 3]

        edge_nearby = self.edge_nearby(ex_features).transpose(-1, -2)
        # [bs, n_points_all, 2]

        meta_type = self.meta_type(ex_features).transpose(-1, -2)
        # [bs, n_points_all, 7]

        edge_nearby_log = F.log_softmax(edge_nearby, dim=-1)
        meta_type_log = F.log_softmax(meta_type, dim=-1)

        return eula_angle, edge_nearby_log, meta_type_log











