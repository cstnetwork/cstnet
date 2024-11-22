import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import matplotlib.pyplot as plt

class PointNetSetAbstractionAllVert(nn.Module):
    def __init__(self, n_center_fps, radius, n_sample_ball, n_channel_in, mlp, is_group_all):
        super().__init__()
        self.n_center_fps = n_center_fps
        self.radius = radius
        self.n_sample_ball = n_sample_ball
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = n_channel_in

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv3d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm3d(out_channel))
            last_channel = out_channel

        self.group_all = is_group_all

    def forward(self, xyz, features):
        # xyz: torch.Size([batch_size, n_points_all, n_points_neighbor, 3])
        # features: torch.Size([batch_size, n_points_all, n_points_neighbor, f])

        new_xyz_list = []
        new_features_list = []
        batch_size = xyz.size()[0]

        for i in range(batch_size):
            if self.group_all:
                if features is None:
                    tmp_fea = None
                else:
                    tmp_fea = features[i, :, :, :]

                new_xyz, new_features = sample_and_group_all_AllVert(xyz[i, :, :, :], tmp_fea)
                new_xyz_list.append(new_xyz.unsqueeze(dim=0))
                new_features_list.append(new_features.unsqueeze(dim=0))

            else:
                if features is None:
                    tmp_fea = None
                else:
                    tmp_fea = features[i, :, :, :]

                new_xyz, new_features = sample_and_group(self.n_center_fps, self.radius, self.n_sample_ball, xyz[i, :, :, :], tmp_fea)

                new_xyz_list.append(new_xyz.unsqueeze(dim=0))
                new_features_list.append(new_features.unsqueeze(dim=0))

        new_xyz = torch.cat(new_xyz_list, dim=0)
        new_features = torch.cat(new_features_list, dim=0)
        # [batch_size, n_points_all, n_center_fps, n_sample_ball, 3 + f]

        new_features = new_features.permute(0, 4, 3, 2, 1).to(torch.float)
        # [batch_size, 3 + f, n_sample_ball, n_center_fps, n_points_all]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_features = F.relu(bn(conv(new_features)))

        # [batch_size, channel_out, n_sample_ball, n_center_fps, n_points_all]
        new_features = torch.max(new_features, 2)[0]
        # [batch_size, channel_out, n_center_fps, n_points_all]

        new_features = new_features.permute(0, 3, 2, 1)
        # [batch_size, n_points_all, n_center_fps, channel_out]

        return new_xyz, new_features
        # new_xyz: [batch_size, n_points_all, n_center_fps, 3]
        # new_features: [batch_size, n_points_all, n_center_fps, channel_out]


def sample_and_group_AllVert(n_center_fps, radius, n_sample_ball, xyz, features, returnfps=False):
    # xyz: torch.Size([batch_size, n_points_all, n_points_neighbor, 3])
    # features: torch.Size([batch_size, n_points_all, n_points_neighbor, f])

    batch_size, n_points_all, n_points_neighbor, n_coor = xyz.shape

    # xyz: torch.Size([batch_size, n_points_all, n_points_neighbor, 3])
    fps_idx = farthest_point_sample_AllVert(xyz, n_center_fps)
    # fps_idx: torch.Size([batch_size, n_points_all, n_center_fps])

    new_xyz = index_points(xyz, fps_idx) # 获取 xyz 中，索引 fps_idx 对应的点
    idx = query_ball_point(radius, n_sample_ball, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(batch_size, n_center_fps, 1, n_coor)

    if features is not None:
        grouped_points = index_points(features, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

    # new_xyz: [batch_size, n_points_all, n_center_fps, 3]
    # new_features: [batch_size, n_points_all, n_center_fps, n_sample_ball, 3 + f]


def sample_and_group_all_AllVert(xyz, points):
    device = xyz.device

    # B: batch_size, N: point number, C: channels
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def farthest_point_sample_AllVert(xyz, n_center_fps):
    device = xyz.device

    batch_size, n_points_all, n_points_neighbor, n_coor = xyz.shape

    centroids = torch.zeros(batch_size, n_points_all, n_center_fps, dtype=torch.long).to(device)

    distance = torch.ones(batch_size, n_points_all, n_points_neighbor).to(device) * 1e10

    farthest = torch.randint(0, n_points_neighbor, (batch_size, n_points_all), dtype=torch.long).to(device)

    batch_indices = torch.arange(n_points_all, dtype=torch.long).to(device)
    batch_indices = torch.tile(batch_indices, (batch_size, 1))

    for i in range(n_center_fps):
        centroids[:, :, i] = farthest

        # print('batch_indices', batch_indices.shape)
        # print('farthest', farthest.shape)
        #
        # print('xyz2', (xyz[0, batch_indices[0, :], farthest[0, :], :].view(n_points_all, 1, 3)).shape)

        centroid_list = []
        for j in range(batch_size):
            centroid = xyz[j, batch_indices[j, :], farthest[j, :], :].view(n_points_all, 1, 3)
            centroid_list.append(centroid.unsqueeze(dim=0))

            # print(centroid.unsqueeze(dim=0).size())

        #     if j == 0:
        #         atupo = centroid.unsqueeze(dim=0)
        #
        #     if j == 1:
        #         btupo = centroid.unsqueeze(dim=0)
        #
        # print('atupo', atupo.size())
        # print('btupo', btupo.size())

        # centroid = torch.cat((atupo, btupo), dim=0)
        centroid = torch.cat(tuple(centroid_list), dim=0)

        # print('xyz', xyz.shape)
        # print('centroid', centroid.shape)

        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask].float()
        farthest = torch.max(distance, -1)[-1]

    return centroids


def index_points_AllVert(points, idx):
    '''
    :param points: torch.Size([batch_size, n_points_all, n_points_neighbor, 3])
    :param idx: torch.Size([batch_size, n_points_all, n_samples])
    :return: torch.Size([batch_size, n_points_all, n_samples, 3])
    '''
    batch_size = points.shape[0]

    new_points_list = []
    for i in range(batch_size):
        new_points_list.append(index_points(points[i, :, :, :], idx[i, :, :]).unsqueeze(dim=0))

    new_points = torch.cat(new_points_list, dim=0)

    return new_points


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


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


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    # xyz: [24, 1024, 3], B: batch_size, N: number of points, C: channels
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    device = xyz.device

    # B: batch_size, N: point number, C: channels
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)

        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            # xyz: torch.Size([24, 1024, 3])
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points normal data, [B, npoint, nsample, C+D]

        new_points = new_points.permute(0, 3, 2, 1).to(torch.float)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module): # sa1
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        # sa1:
        # npoint=512
        # radius_list = [0.1, 0.2, 0.4]
        # nsample_list = [32, 64, 128]
        # in_channel=3
        # mlp_list = [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint # sa1, 512
        self.radius_list = radius_list # sa1, [0.1, 0.2, 0.4]
        self.nsample_list = nsample_list # sa1, [32, 64, 128]
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3 # sa1, 6
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
        # self.conv_blocks = nn.Conv2d(6, 32, 32, 64), nn.Conv2d(6, 64, 64, 128), nn.Conv2d(6, 64, 96, 128)
        # self.bn_blocks = nn.BatchNorm2d(32, 32, 64), nn.BatchNorm2d(64, 64, 128), nn.BatchNorm2d(64, 96, 128)

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        # sa1: xyz = points: Size([16, 2048, 3])

        # B: batch_size, N: npoints, C: channel
        B, N, C = xyz.shape

        # S: sample, sa1: 512
        S = self.npoint

        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))

        new_points_list = []

        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]

            # sa1: torch.Size([16, 512, 32])
            group_idx = query_ball_point(radius, K, xyz, new_xyz)

            # torch.Size([16, 512, 32, 3])
            grouped_xyz = index_points(xyz, group_idx)

            # new_xyz: torch.Size([B, S, C]) sa1: Size([16, 512, 3])
            grouped_xyz -= new_xyz.view(B, S, 1, C)

            # sa1: xyz = points: Size([16, 3, 2048])
            if points is not None:
                grouped_points = index_points(points, group_idx)

                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))

            new_points = torch.max(grouped_points, 2)[0] # [B, D', S]

            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # sa1: torch.Size([16, 3, 512])

        new_points_concat = torch.cat(new_points_list, dim=1)
        # sa1: torch.Size([16, 64 + 128 + 128, 512])
        # sa1: 16 batch_size, 64 + 128 + 128 multiple scale feature, 512: number of centroids

        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp): # fp1: in_channel=150, mlp=[128, 128]
        super(PointNetFeaturePropagation, self).__init__()
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


class PARA_FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp): # fp1: in_channel=150, mlp=[128, 128]
        super(PARA_FeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        # self.mlp_convs = nn.Conv1d(150,128), nn.Conv1d(128,128)
        # self.mlp_bns = nn.BatchNorm1d(128), nn.BatchNorm1d(128)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: torch.Size([batch_size, 3, n_points]) 原始点云坐标数据 fp1: torch.Size([16, 3, 2048])
        xyz2: torch.Size([batch_size, 3, n_samples]) 从原始点云中采样获得的点数据 fp1: torch.Size([16, 3, 512])
        points1: torch.Size([16, 22, 2048])
        points2: torch.Size([16, 128, 512])

        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """

        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # xyz1: Size([16, 2048, 3])   (fp1)
        # xyz2: Size([16, 512, 3])   (fp1)

        points2 = points2.permute(0, 2, 1)
        # points2： torch.Size([16, 512, 128])   (fp1)

        # B: batch_size, N: n_points, C: channel(3), S: n_samples(512)
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
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        # interpolated_points: torch.Size([16, 2048, 128])

        # points1: torch.Size([16, 22, 2048]), 22 = 16(all_object_class) + 3(xyz) + 3(xyz)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
            # new_points: torch.Size([16, 2048, 150=22+128])
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        # new_points: torch.Size([16, 150=22+128, 2048])

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


def test_farthest_point_sample_AllVert():
    test_tensor = torch.rand((2, 1000, 50, 3))
    fps_idx = farthest_point_sample_AllVert(test_tensor, 25)
    print(fps_idx.size())

    fps_idx2 = farthest_point_sample(test_tensor[:, 0, :, :], 25)
    print(fps_idx2.size())

    print(fps_idx[0, 0, :] - fps_idx2[0, :])

    points1 = test_tensor[0, 0, fps_idx[0, 0, :], :].numpy()
    points2 = test_tensor[0, 0, fps_idx2[0, :], :].numpy()

    fig = plt.figure()

    ax1 = fig.add_subplot(121, projection='3d')

    ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2])

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points2[:, 0], points2[:, 1], points2[:, 2])

    plt.show()

def test_index_points_AllVert():
    test_tensor = torch.rand((2, 1000, 50, 3))
    test_tensor_ind = torch.randint(0, 50, (2, 1000, 25))

    # print(index_points_AllVert(test_tensor[:, 0, :, :], test_tensor_ind[:, 0, :]).size())
    print(index_points_AllVert(test_tensor, test_tensor_ind).size())


def test_allvert():
    test_tensor = torch.rand((2, 500, 50, 3)).cuda()

    anet = PointNetSetAbstractionAllVert(100, 0.2, 32, 3, [64, 64, 128], False).cuda()

    print(anet(test_tensor, None)[0].size())
    print(anet(test_tensor, None)[1].size())


if __name__ == '__main__':
    # test_allvert()

    points_all = torch.rand(2, 500, 3)
    tmp_all = points_all[0, :, :]
    tmp_all = torch.tile(tmp_all, (500, 1, 1))

    print(tmp_all.size())


