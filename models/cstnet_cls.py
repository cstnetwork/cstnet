
"""
测试得到的效果最好的会放在这个文件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


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


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    返回局部点索引                   torch.Size([batch_size, n_centroid,  n_local_sample])   sa1: torch.Size([16, 512, 32])
    xyz：    点云中全部点            torch.Size([batch_size, n_all_point, channel])          sa1: torch.Size([16, 2048, 3])
    new_xyz：点云中当作采样球心的点    torch.Size([batch_size, n_centroid,  channel])          sa1: torch.Size([16, 512, 3])
    radius： 采样局部区域半径
    nsample：每个局部区域最大采样点数

    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
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


def index_points(points, idx):
    """
    返回 points 中 索引 idx 对应的点
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
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
    """
    最远采样法进行采样，返回采样点的索引
    Input:
        xyz: pointcloud data, [batch_size, n_points_all, 3]
        n_samples: number of samples
    Return:
        centroids: sampled pointcloud index, [batch_size, n_samples]
    """
    device = xyz.device

    # xyz: [24, 1024, 3], B: batch_size, N: number of points, C: channels
    B, N, C = xyz.shape

    # 生成 B 行，n_samples 列的全为零的矩阵
    centroids = torch.zeros(B, n_samples, dtype=torch.long).to(device)

    # 生成 B 行，N 列的矩阵，每个元素为 1e10
    distance = torch.ones(B, N).to(device) * 1e10

    # 生成随机整数tensor，整数范围在[0，N)之间，包含0不包含N，矩阵各维度长度必须用元组传入，因此写成(B,)
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    # 生成 [0, B) 整数序列
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


def onehot_merge(tensor1, tensor2):
    '''
    合并两个 onehot, 在最后一维上合并
    :param tensor1: [bs, n_pnts, n_near, channel1]
    :param tensor2: [bs, n_pnts, n_near, channel2]
    :return: [bs, n_pnts, n_near, channel1 * channel2]
    '''
    channel1 = tensor1.size()[-1]

    tensor_all = []
    for i in range(channel1):
        tensor_all.append(tensor1[:, :, :, i].unsqueeze(-1) * tensor2)

    return torch.cat(tensor_all, dim=-1)


def sample_and_group(n_center, n_near, xyz, eula_angle, edge_nearby, meta_type, fea):
    """
    采样并以采样点为圆心集群，使用knn
    Input:
        npoint: 最远采样法的采样点数，即集群数, 为None则不采样
        radius: 集群过程中的半径
        nsample: 每个集群中的点数
    Return:
        center_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_fea: sampled points data, [B, npoint, nsample, 3+D]
    """
    # xyz: [24, 1024, 3], B: batch_size, N: number of points, C: channels

    # idx_surfknn_all = utils.surface_knn(xyz, n_near, 10)
    idx_surfknn_all = utils.get_neighbor_index(xyz, n_near)

    if n_center is None:
        center_xyz = xyz
        canter_eula = eula_angle
        center_near = edge_nearby
        center_meta = meta_type

        idx = idx_surfknn_all

    else:
        fps_idx = farthest_point_sample(xyz, n_center)  # 采样后的点索引 troch.size([B, npoint])

        center_xyz = index_points(xyz, fps_idx)  # 获取 xyz 中，索引 fps_idx 对应的点
        canter_eula = index_points(eula_angle, fps_idx)
        center_near = index_points(edge_nearby, fps_idx)
        center_meta = index_points(meta_type, fps_idx)

        idx = index_points(idx_surfknn_all, fps_idx)

    g_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
    g_eula = index_points(eula_angle, idx)
    g_near = index_points(edge_nearby, idx)
    g_meta = index_points(meta_type, idx)

    g_xyz_relative = g_xyz - center_xyz.unsqueeze(2)  # 周围点到中心点的向量
    g_eula_relative = g_eula - canter_eula.unsqueeze(2)

    g_near_cat = torch.cat([g_near, center_near.unsqueeze(2).repeat(1, 1, n_near, 1)], dim=-1)
    g_meta_cat = torch.cat([g_meta, center_meta.unsqueeze(2).repeat(1, 1, n_near, 1)], dim=-1)

    g_fea = index_points(fea, idx)
    g_fea = torch.cat([g_xyz_relative, g_eula_relative, g_near_cat, g_meta_cat, g_fea], dim=-1)

    if n_center is None:
        center_fea = fea
    else:
        center_fea = index_points(fea, fps_idx)

    return center_xyz, canter_eula, center_near, center_meta, center_fea, g_fea


def sample_and_group_all(xyz, eula_angle, edge_nearby, meta_type, fea):
    """
    返回 0：全为零的 tensor，shape 为 [batch_size, 1, channels]，仅用作占位，因为返回的最后一层中心点没用
    返回 1：先把xyz view成[B, 1, N, C]，xyz输入时是[B, N, C]，然后返回(如果points为none的话)

    相当于把输入的所有xyz，当成一个点的邻近点
    """
    # B: batch_size, N: point number, C: channels
    B, N, C = xyz.shape

    g_xyz = xyz.view(B, 1, N, C)
    g_eula = eula_angle.view(B, 1, N, -1)

    # g_near = onehot_merge(edge_nearby.view(B, 1, N, -1), edge_nearby.view(B, 1, N, -1))
    # g_meta = onehot_merge(meta_type.view(B, 1, N, -1), meta_type.view(B, 1, N, -1))

    g_near = edge_nearby.view(B, 1, N, -1).repeat(1, 1, 1, 2)
    g_meta = meta_type.view(B, 1, N, -1).repeat(1, 1, 1, 2)

    if fea is not None:
        new_fea = torch.cat([g_xyz, g_eula, g_near, g_meta, fea.view(B, 1, N, -1)], dim=-1)
    else:
        new_fea = torch.cat([g_xyz, g_eula, g_near, g_meta], dim=-1)

    return None, None, None, None, None, new_fea


class AttentionInPnts(nn.Module):
    '''
    点之间的注意力机制
    '''
    def __init__(self, channel_in):
        super().__init__()

        self.fai = utils.full_connected_conv2d([channel_in, channel_in + 8, channel_in])
        self.psi = utils.full_connected_conv2d([channel_in, channel_in + 8, channel_in])
        self.alpha = utils.full_connected_conv2d([channel_in, channel_in + 8, channel_in])
        self.gamma = utils.full_connected_conv2d([channel_in, channel_in + 8, channel_in])

    def forward(self, x_i, x_j):
        # x_i: [bs, channel, 1, n_point]
        # x_j: [bs, channel, n_near, n_point]
        # p_i: [bs, 3, 1, n_point]
        # p_j: [bs, 3, n_near, n_point]

        bs, channel, n_near, n_point = x_j.size()

        fai_xi = self.fai(x_i)  # -> [bs, channel, 1, npoint]
        psi_xj = self.psi(x_j)  # -> [bs, channel, n_near, npoint]
        alpha_xj = self.alpha(x_j)  # -> [bs, channel, n_near, npoint]

        y_i = (channel * F.softmax(self.gamma(fai_xi - psi_xj), dim=1)) * alpha_xj  # -> [bs, channel, n_near, npoint]
        y_i = torch.sum(y_i, dim=2)  # -> [bs, channel, npoint]
        y_i = y_i / n_near  # -> [bs, channel, npoint]
        y_i = y_i.permute(0, 2, 1)  # -> [bs, npoint, channel]

        return y_i


class SetAbstraction(nn.Module):
    """
    set abstraction 层
    包含sampling、grouping、PointNet层
    """
    def __init__(self, n_center, n_near, in_channel, mlp, group_all):
        '''
        :param n_center: 使用FPS查找的中心点数，因此该点数也是池化到的点数
        :param radius: 沿每个中心点进行 ball query 的半径
        :param n_near: 每个ball里的点数最大值，感觉就是查找这个数目的点，和半径无关
        :param in_channel: 输入特征维度
        :param mlp: list，表示最后接上的 MLP 各层维度
        :param group_all: 是否将全部特征集中到一个点
        '''
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
        self.group_all = group_all

        self.attention_points = AttentionInPnts(last_channel)

    def forward(self, xyz, eula_angle, edge_nearby, meta_type, fea=None):
        """
        Input:
            xyz: 点的 xyz 特征。input points position data, [B, C, N]
            points: 点的 ijk 特征。input points data, [B, D, N]
        Return:
            new_xyz: 处理后的 xyz 特征。sampled points position data, [B, C, S]
            new_points_concat: 处理后的 ijk 特征。sample points feature data, [B, D', S]
        """
        # xyz: [bs, n_point, 3]
        # eula_angle: [bs, n_point, 3]
        # edge_nearby: [bs, n_point, 2] one-hot
        # meta_type: [bs, n_point, 4] one-hot
        # fea: [bs, n_point, n_channel] one-hot

        if self.group_all:
            center_xyz, center_eula, center_near, center_meta, center_fea, new_fea = sample_and_group_all(xyz, eula_angle, edge_nearby, meta_type, fea)
        else:
            # xyz: torch.Size([24, 1024, 3])
            center_xyz, center_eula, center_near, center_meta, center_fea, new_fea = sample_and_group(self.n_center, self.n_near, xyz, eula_angle, edge_nearby, meta_type, fea)

        new_fea = new_fea.permute(0, 3, 2, 1)  # [bs, emb, n_near, n_point]

        if not self.group_all:
            xyz_for_attention = torch.zeros_like(center_xyz, dtype=torch.float)
            euler_for_attention = torch.zeros_like(center_eula, dtype=torch.float)
            near_for_attention = center_near.repeat(1, 1, 2)
            meta_for_attention = center_meta.repeat(1, 1, 2)
            center_fea_for_attention = torch.cat([xyz_for_attention, euler_for_attention, near_for_attention, meta_for_attention, center_fea], dim=-1).unsqueeze(2).permute(0, 3, 2, 1)

        # center_fea_for_attention = center_fea.unsqueeze(2).permute(0, 3, 2, 1)  # [bs, emb, 1, n_point]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_fea = F.relu(bn(conv(new_fea)))

            if not self.group_all:
                center_fea_for_attention = F.relu(bn(conv(center_fea_for_attention)))

        if self.group_all:
            # 每个采样点的邻近点的特征维度取最大值
            new_fea = torch.max(new_fea, 2)[0]
            new_fea = new_fea.permute(0, 2, 1)
        else:
            # 使用点之间的注意力机制更新特征
            new_fea = self.attention_points(center_fea_for_attention, new_fea)

        # 使用残差形式，将输入短接到输出
        if center_fea is not None:
            new_fea = torch.cat([center_fea, new_fea], dim=-1)

        return center_xyz, center_eula, center_near, center_meta, new_fea


class CrossAttention_Cls(nn.Module):
    '''
    CrossAttention分类模型
    '''
    def __init__(self, n_classes, n_metatype):
        super().__init__()

        self.SA_ChannelOut = 512 + 256
        # 输入为 xyz:3, eula:3, near_cat:2*2, meta_cat:4*2
        in_channel = 3 + 3 + 2*2 + n_metatype*2

        self.preprocess = utils.full_connected_conv1d([3+3+2+n_metatype, 16, 32])

        self.sa1 = SetAbstraction(n_center=1024, n_near=32, in_channel=(32+in_channel), mlp=[64, 64+8, 64+16], group_all=False)
        self.sa2 = SetAbstraction(n_center=512, n_near=32, in_channel=(32+in_channel) + (64+16), mlp=[64+32, 64+32+16, 128], group_all=False)
        self.sa3 = SetAbstraction(n_center=128, n_near=64, in_channel=(32+in_channel+64+16) + 128, mlp=[128+32, 128+32+16, 128+64], group_all=False)
        self.sa4 = SetAbstraction(n_center=64, n_near=64, in_channel=(32+in_channel+64+16+128) + (128+64), mlp=[256, 256+64, 256+128], group_all=False)
        self.sa5 = SetAbstraction(n_center=None, n_near=None, in_channel=(32+in_channel+64+16+128+128+64) + (256+128), mlp=[512, 512+128, self.SA_ChannelOut], group_all=True)

        self.fc1 = nn.Linear(self.SA_ChannelOut, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, n_classes)

    def forward(self, xyz, eula_angle, edge_nearby, meta_type):
        batch_size, _, _ = xyz.shape

        orig_fea = torch.cat([xyz, eula_angle, edge_nearby, meta_type], dim=-1).transpose(1, 2)
        orig_fea = self.preprocess(orig_fea).transpose(1, 2)

        l1_xyz, l1_eula, l1_near, l1_meta, l1_fea = self.sa1(xyz, eula_angle, edge_nearby, meta_type, orig_fea)
        l2_xyz, l2_eula, l2_near, l2_meta, l2_fea = self.sa2(l1_xyz, l1_eula, l1_near, l1_meta, l1_fea)
        l3_xyz, l3_eula, l3_near, l3_meta, l3_fea = self.sa3(l2_xyz, l2_eula, l2_near, l2_meta, l2_fea)
        l4_xyz, l4_eula, l4_near, l4_meta, l4_fea = self.sa4(l3_xyz, l3_eula, l3_near, l3_meta, l3_fea)
        _, _, _, _,                        l5_fea = self.sa5(l4_xyz, l4_eula, l4_near, l4_meta, l4_fea)

        x = l5_fea.view(batch_size, self.SA_ChannelOut)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x


if __name__ == '__main__':
    xyz_tensor = torch.rand(2, 2500, 3).cuda()
    eula_tensor = torch.rand(2, 2500, 3).cuda()
    edge_tensor = torch.rand(2, 2500, 2).cuda()
    meta_tensor = torch.rand(2, 2500, 4).cuda()

    anet = CrossAttentionNet_Cls(10, 4).cuda()

    pred = anet(xyz_tensor, eula_tensor, edge_tensor, meta_tensor)

    print(pred.shape)
    print(pred)







