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


def onehot_merge(tensor1, tensor2):
    channel1 = tensor1.size()[-1]

    tensor_all = []
    for i in range(channel1):
        tensor_all.append(tensor1[:, :, :, i].unsqueeze(-1) * tensor2)

    return torch.cat(tensor_all, dim=-1)


def sample_and_group(n_center, n_near, xyz, eula_angle, edge_nearby, meta_type, fea):
    idx_surfknn_all = utils.get_neighbor_index(xyz, n_near)

    if n_center is None:
        center_xyz = xyz
        canter_eula = eula_angle
        center_near = edge_nearby
        center_meta = meta_type

        idx = idx_surfknn_all

    else:
        fps_idx = farthest_point_sample(xyz, n_center)

        center_xyz = index_points(xyz, fps_idx)
        canter_eula = index_points(eula_angle, fps_idx)
        center_near = index_points(edge_nearby, fps_idx)
        center_meta = index_points(meta_type, fps_idx)

        idx = index_points(idx_surfknn_all, fps_idx)

    g_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
    g_eula = index_points(eula_angle, idx)
    g_near = index_points(edge_nearby, idx)
    g_meta = index_points(meta_type, idx)

    g_xyz_relative = g_xyz - center_xyz.unsqueeze(2)
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
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(xyz.device)

    g_xyz = xyz.view(B, 1, N, C)
    g_eula = eula_angle.view(B, 1, N, -1)

    # g_near = onehot_merge(edge_nearby.view(B, 1, N, -1), edge_nearby.view(B, 1, N, -1))
    # g_meta = onehot_merge(meta_type.view(B, 1, N, -1), meta_type.view(B, 1, N, -1))

    g_near = edge_nearby.view(B, 1, N, -1).repeat(1, 1, 1, 2)
    g_meta = meta_type.view(B, 1, N, -1).repeat(1, 1, 1, 2)

    center_fea = torch.max(fea, dim=1, keepdim=True)[0]

    if fea is not None:
        new_fea = torch.cat([g_xyz, g_eula, g_near, g_meta, fea.view(B, 1, N, -1)], dim=-1)
    else:
        new_fea = torch.cat([g_xyz, g_eula, g_near, g_meta], dim=-1)

    return new_xyz, None, None, None, center_fea, new_fea


class AttentionInPnts(nn.Module):
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
    def __init__(self, n_center, n_near, in_channel, mlp, group_all):
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
        self.group_all = group_all

        self.attention_points = AttentionInPnts(last_channel)

    def forward(self, xyz, eula_angle, edge_nearby, meta_type, fea=None):
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
            new_fea = torch.max(new_fea, 2)[0]
            new_fea = new_fea.permute(0, 2, 1)
        else:
            new_fea = self.attention_points(center_fea_for_attention, new_fea)

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
            # points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = new_points.permute(0, 2, 1)
        return new_points


class CrossAttention_Seg(nn.Module):
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

    anet = CrossAttention_Seg(10, 4).cuda()

    pred = anet(xyz_tensor, eula_tensor, edge_tensor, meta_tensor)

    print(pred.shape)
    print(pred)

