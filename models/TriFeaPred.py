import torch.nn as nn
import torch
import torch.nn.functional as F

from utils import full_connected_conv1d, full_connected
from pointnet2_utils import index_points


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


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def sample_and_group(n_near, xyz, fea):
    idx_knn_all = knn(xyz, n_near)
    grouped_xyz = index_points(xyz, idx_knn_all)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - xyz.unsqueeze(2)

    if fea is not None:
        grouped_fea = index_points(fea, idx_knn_all)
        new_fea = torch.cat([grouped_xyz_norm, grouped_fea], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_fea = grouped_xyz_norm

    return new_fea


class SetAbstraction(nn.Module):
    def __init__(self, n_near, in_channel, mlp):
        super().__init__()

        self.n_near = n_near
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        new_points = sample_and_group(self.n_near, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1).to(torch.float)  # [B, C+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)

        return new_points


class FeaPropagate(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, points1, points2):
        new_points = torch.cat([points1, points2], dim=-1)
        new_points = new_points.permute(0, 2, 1)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = new_points.permute(0, 2, 1)

        return new_points


class TriFeaPred(nn.Module):
    def __init__(self, n_metatype, n_embout=256, n_neighbor=100):
        super().__init__()

        self.n_neighbor = n_neighbor

        prep_channel = 16
        self.preprocess = full_connected_conv1d([3, 8, prep_channel])

        self.sa1 = SetAbstraction(n_near=16, in_channel=prep_channel+3, mlp=[32, 32+8, 32+16])
        self.sa2 = SetAbstraction(n_near=24, in_channel=(32+16)+3, mlp=[32+16+8, 64, 64+16])
        self.sa3 = SetAbstraction(n_near=32, in_channel=(64+16)+3, mlp=[64+16+8, 64+32, 64+32+16])
        self.sa4 = SetAbstraction(n_near=24, in_channel=(64+32+16)+3, mlp=[64+32+16+8, 128, 128+32])
        self.sa5 = SetAbstraction(n_near=16, in_channel=(128+32)+3, mlp=[128+32+16, 128+64, 256])

        self.fp5 = FeaPropagate(in_channel=256+(128+32), mlp=[256+128, 256+64+32, 256+64])
        self.fp4 = FeaPropagate(in_channel=(64+32+16)+(256+64), mlp=[256+128, 256+128+64, 256+128])
        self.fp3 = FeaPropagate(in_channel=(64+16)+(256+128), mlp=[256+128+64, 256+128+32+16, 256+128+32])
        self.fp2 = FeaPropagate(in_channel=(32+16)+(256+128+32), mlp=[256+128+64+16+8, 256+128+64+16, 256+128+64])
        self.fp1 = FeaPropagate(in_channel=3+prep_channel+(256+128+64), mlp=[256+128+64+32, 256+128+64+32+16, 512])

        self.conv1 = nn.Conv1d(512, 512-128, 1)
        self.bn1 = nn.BatchNorm1d(512-128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(512-128, n_embout, 1)
        self.bn2 = nn.BatchNorm1d(n_embout)
        self.drop2 = nn.Dropout(0.5)

        eula_div = (n_embout / 3) ** 0.25
        self.eula_angle = full_connected_conv1d([n_embout, int(n_embout/eula_div), int(n_embout/eula_div**2), int(n_embout/eula_div**3), 3])

        near_div = (n_embout / 2) ** 0.25
        self.edge_nearby = full_connected_conv1d([n_embout, int(n_embout/near_div), int(n_embout/near_div**2), int(n_embout/near_div**3), 2])

        eula_div = (n_embout / n_metatype) ** 0.25
        self.meta_type = full_connected_conv1d([n_embout, int(n_embout/eula_div), int(n_embout/eula_div**2), int(n_embout/eula_div**3), n_metatype])

    def forward(self, xyz):
        # -> xyz: [bs, n_points, 3]

        # Set Abstraction layers
        l0_fea = self.preprocess(xyz.transpose(1, 2)).transpose(1, 2)  # -> [bs, n_point, emb]

        l1_fea = self.sa1(xyz, l0_fea)  # -> [bs, n_point, emb]
        l2_fea = self.sa2(xyz, l1_fea)  # -> [bs, n_point, emb]
        l3_fea = self.sa3(xyz, l2_fea)
        l4_fea = self.sa4(xyz, l3_fea)
        l5_fea = self.sa5(xyz, l4_fea)

        # Feature Propagation layers
        l4_fea = self.fp5(l4_fea, l5_fea)  # -> [bs, n_point, emb]
        l3_fea = self.fp4(l3_fea, l4_fea)  # -> [bs, n_point, emb]
        l2_fea = self.fp3(l2_fea, l3_fea)  # -> [bs, n_point, emb]
        l1_fea = self.fp2(l1_fea, l2_fea)  # -> [bs, n_point, emb]
        l0_fea = self.fp1(torch.cat([xyz, l0_fea], dim=-1), l1_fea)  # -> [bs, n_point, emb]

        fea = self.drop1(F.relu(self.bn1(self.conv1(l0_fea.permute(0, 2, 1)))))
        fea = self.drop2(F.relu(self.bn2(self.conv2(fea))))

        eula_angle = self.eula_angle(fea).transpose(-1, -2)
        # [bs, n_points_all, 3]

        edge_nearby = self.edge_nearby(fea).transpose(-1, -2)
        # [bs, n_points_all, 2]

        meta_type = self.meta_type(fea).transpose(-1, -2)
        # [bs, n_points_all, 7]

        edge_nearby_log = F.log_softmax(edge_nearby, dim=-1)
        meta_type_log = F.log_softmax(meta_type, dim=-1)

        return eula_angle, edge_nearby_log, meta_type_log


if __name__ == '__main__':
    input_xyz = torch.rand([2, 2500, 3])

    predictor = TriFeaPred(4)

    pred_eula, pred_near, pred_meta = predictor(input_xyz)

    print(pred_eula.size())
    print(pred_near.size())
    print(pred_meta.size())
