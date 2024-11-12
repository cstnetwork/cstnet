
"""
分割模型
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
    new_xyz = torch.zeros(B, 1, C).to(xyz.device)

    g_xyz = xyz.view(B, 1, N, C)
    g_eula = eula_angle.view(B, 1, N, -1)

    # g_near = onehot_merge(edge_nearby.view(B, 1, N, -1), edge_nearby.view(B, 1, N, -1))
    # g_meta = onehot_merge(meta_type.view(B, 1, N, -1), meta_type.view(B, 1, N, -1))

    g_near = edge_nearby.view(B, 1, N, -1).repeat(1, 1, 1, 2)
    g_meta = meta_type.view(B, 1, N, -1).repeat(1, 1, 1, 2)

    # 中心点特征定义为最大池化获得的特征
    center_fea = torch.max(fea, dim=1, keepdim=True)[0]

    if fea is not None:
        new_fea = torch.cat([g_xyz, g_eula, g_near, g_meta, fea.view(B, 1, N, -1)], dim=-1)
    else:
        new_fea = torch.cat([g_xyz, g_eula, g_near, g_meta], dim=-1)

    return new_xyz, None, None, None, center_fea, new_fea


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


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp): # fp1: in_channel=150, mlp=[128, 128]
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz2为从xyz1中采样获得的点坐标，points1, points2 为对应属性
        对于xyz1中的某个点(center)，找到xyz2中与之最近的3个点(nears)，将nears的特征进行加权和，得到center的插值特征
        nears中第i个点(near_i)特征的权重为 [1/d(near_i)]/sum(k=1->3)[1/d(near_k)]
        d(near_i)为 center到near_i的距离，即距离越近，权重越大
        之后拼接points1与xyz中每个点的更新属性，再利用MLP对每个点的特征单独进行处理

        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: sampled points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)
        # points2 = points2.permute(0, 2, 1)

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # 计算xyz1中的每个点到xyz2中每个点的距离 xyz1:[bs, N, 3], xyz2:[bs, S, 3], return: [bs, N, S]
            dists = square_distance(xyz1, xyz2)

            # 计算每个初始点到采样点距离最近的3个点，sort 默认升序排列
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            # 最近距离的每行求倒数
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)

            # 求倒数后每行中每个数除以该行之和
            weight = dist_recip / norm  # ->[B, N, 3]

            # index_points(points2, idx): 为原始点集中的每个点找到采样点集中与之最近的3个三个点的特征 -> [B, N, 3, D]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            # skip link concatenation
            # points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        # 使用MLP对每个点的特征单独进行处理
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = new_points.permute(0, 2, 1)
        return new_points


class CrossAttention_Seg(nn.Module):
    '''
    CrossAttentionNet分类模型
    '''
    def __init__(self, n_classes, n_metatype):
        super().__init__()
        print('param segmentation net')

        self.SA_ChannelOut = 512 + 256
        # 输入为 xyz:3, eula:3, near_cat:2*2, meta_cat:4*2
        in_channel = 3 + 3 + 2*2 + n_metatype*2

        self.preprocess = utils.full_connected_conv1d([3+3+2+n_metatype, 16, 32])

        self.sa1 = SetAbstraction(n_center=1024, n_near=32, in_channel=(32+in_channel), mlp=[64, 64+8, 64+16], group_all=False)
        self.sa2 = SetAbstraction(n_center=512, n_near=32, in_channel=(32+in_channel) + (64+16), mlp=[64+32, 64+32+16, 128], group_all=False)
        self.sa3 = SetAbstraction(n_center=128, n_near=64, in_channel=(32+in_channel+64+16) + 128, mlp=[128+32, 128+32+16, 128+64], group_all=False)
        self.sa4 = SetAbstraction(n_center=64, n_near=64, in_channel=(32+in_channel+64+16+128) + (128+64), mlp=[256, 256+64, 256+128], group_all=False)
        self.sa5 = SetAbstraction(n_center=None, n_near=None, in_channel=(32+in_channel+64+16+128+128+64) + (256+128), mlp=[512, 512+128, self.SA_ChannelOut], group_all=True)

        self.fp5 = FeaturePropagation(in_channel=1584 + 816, mlp=[self.SA_ChannelOut, 640])  # in_chanell = points2_chanell + points1_channel
        self.fp4 = FeaturePropagation(in_channel=640 + 432, mlp=[512 + 64, 512 + 32])  # in_chanell = points2_chanell + points1_channel
        self.fp3 = FeaturePropagation(in_channel=544 + 240, mlp=[512, 256 + 128 + 64 + 32])  # in_chanell = points2_chanell + points1_channel
        self.fp2 = FeaturePropagation(in_channel=480 + 112, mlp=[256 + 128, 256 + 64])
        self.fp1 = FeaturePropagation(in_channel=320 + 32, mlp=[256 + 32, 256])

        self.afterprocess = utils.full_connected_conv1d([256, 128, n_classes])

    def forward(self, xyz, eula_angle, edge_nearby, meta_type):
        batch_size, _, _ = xyz.shape

        orig_fea = torch.cat([xyz, eula_angle, edge_nearby, meta_type], dim=-1).transpose(1, 2)
        orig_fea = self.preprocess(orig_fea).transpose(1, 2)

        l1_xyz, l1_eula, l1_near, l1_meta, l1_fea = self.sa1(xyz, eula_angle, edge_nearby, meta_type, orig_fea)
        l2_xyz, l2_eula, l2_near, l2_meta, l2_fea = self.sa2(l1_xyz, l1_eula, l1_near, l1_meta, l1_fea)
        l3_xyz, l3_eula, l3_near, l3_meta, l3_fea = self.sa3(l2_xyz, l2_eula, l2_near, l2_meta, l2_fea)
        l4_xyz, l4_eula, l4_near, l4_meta, l4_fea = self.sa4(l3_xyz, l3_eula, l3_near, l3_meta, l3_fea)
        l5_xyz, _, _, _,                   l5_fea = self.sa5(l4_xyz, l4_eula, l4_near, l4_meta, l4_fea)

        l4_fea = self.fp5(l4_xyz, l5_xyz, l4_fea, l5_fea)
        l3_fea = self.fp4(l3_xyz, l4_xyz, l3_fea, l4_fea)
        l2_fea = self.fp3(l2_xyz, l3_xyz, l2_fea, l3_fea)
        l1_fea = self.fp2(l1_xyz, l2_xyz, l1_fea, l2_fea)
        l0_fea = self.fp1(xyz, l1_xyz, orig_fea, l1_fea)

        feat = self.afterprocess(l0_fea.permute(0, 2, 1))
        feat = F.log_softmax(feat.permute(0, 2, 1), dim=-1)

        return feat


if __name__ == '__main__':
    xyz_tensor = torch.rand(2, 2500, 3).cuda()
    eula_tensor = torch.rand(2, 2500, 3).cuda()
    edge_tensor = torch.rand(2, 2500, 2).cuda()
    meta_tensor = torch.rand(2, 2500, 4).cuda()

    anet = CrossAttentionNet_Seg(10, 4).cuda()

    pred = anet(xyz_tensor, eula_tensor, edge_tensor, meta_tensor)

    print(pred.shape)
    print(pred)

