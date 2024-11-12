'''
修改版的
使用点之间注意力机制更新特征
'''
# 工具包
import torch.nn as nn
import torch
import torch.nn.functional as F

# 自定义模块
from utils import full_connected_conv1d, full_connected_conv2d
from pointnet2_utils import index_points




def knn(vertices: "(bs, vertice_num, 3)",  neighbor_num: int, is_backdis: bool = False):
    """
    获取每个点最近的k个点的索引
    Return: (bs, vertice_num, neighbor_num)
    """
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


def sample_and_group(n_near, xyz, fea):
    """
    采样并以采样点为圆心集群，使用knn
    Input:
        npoint: 最远采样法的采样点数，即集群数, 为None则不采样
        radius: 集群过程中的半径
        nsample: 每个集群中的点数
        xyz: input points position data, [B, N, 3] 点坐标
        points: input points data, [B, N, D] 法向量
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    # xyz: [24, 1024, 3], B: batch_size, N: number of points, C: channels

    idx_knn_all = knn(xyz, n_near)
    grouped_xyz = index_points(xyz, idx_knn_all)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - xyz.unsqueeze(2)

    grouped_fea = index_points(fea, idx_knn_all)
    new_fea = torch.cat([grouped_xyz_norm, grouped_fea], dim=-1)  # [B, npoint, nsample, C+D]

    return new_fea


class AttentionInPnts(nn.Module):
    '''
    点之间的注意力机制
    '''
    def __init__(self, channel_in, channel_qk=64):
        super().__init__()

        self.sqrt_qk = (channel_in // 2) ** 0.5

        self.WQ = full_connected_conv2d([channel_in, channel_in // 2], bias=False)
        self.WK = full_connected_conv2d([channel_in, channel_in // 2], bias=False)
        self.WV = full_connected_conv2d([channel_in, channel_in], bias=False)

    def forward(self, fea_center, fea_near):
        # fea_center: [bs, n_point, 1, channel_in]
        # fea_near: [bs, n_point, n_near, channel_in]

        # 先将其转换为统一的格式
        fea_near = fea_near.permute(0, 3, 2, 1)  # -> [bs, channel, n_near, npoint]
        fea_center = fea_center.permute(0, 3, 2, 1)  # -> [bs, channel, 1, npoint]
        bs, channel, _, n_points = fea_center.size()

        # 使用点之间的注意力机制更新特征
        MK_neighbor = self.WK(fea_near)  # -> [bs, channel_qk, n_near, npoint]
        MV_neighbor = self.WV(fea_near)  # -> [bs, channel_v, n_near, npoint]

        MQ_center = self.WQ(fea_center)  # -> [bs, channel_qk, 1, npoint]
        MK_center = self.WK(fea_center)  # -> [bs, channel_qk, 1, npoint]
        MV_center = self.WV(fea_center)  # -> [bs, channel_v, 1, npoint]

        weight_center_near = torch.sum(MQ_center * MK_neighbor, dim=1, keepdim=True) / self.sqrt_qk  # -> [bs, 1, n_near, npoint]
        weight_center_center = torch.sum(MQ_center * MK_center, dim=1, keepdim=True) / self.sqrt_qk  # -> [bs, 1, 1, npoint]

        weight_all = torch.cat([weight_center_near, weight_center_center], dim=2)  # -> [bs, 1, n_near+1, npoint]
        weight_all = F.softmax(weight_all, dim=2)  # -> [bs, 1, n_near+1, npoint]
        value_all = torch.cat([MV_neighbor, MV_center], dim=2)  # -> [bs, channel_v, n_near+1, npoint]

        weighted_values = weight_all * value_all  # -> [bs, channel_v, n_near+1, npoint]
        weighted_values = torch.sum(weighted_values, dim=2)  # -> [bs, channel_v, npoint]

        weighted_values = weighted_values.permute(0, 2, 1)  # -> [bs, npoint, channel_v]
        return weighted_values


class SetAbstraction(nn.Module):
    """
    更改后，不 sample
    """
    def __init__(self, n_near, in_channel, mlp):
        '''
        :param npoint: 使用FPS查找的中心点数，因此该点数也是池化到的点数
        :param radius: 沿每个中心点进行 ball query 的半径
        :param nsample: 每个ball里的点数最大值，感觉就是查找这个数目的点，和半径无关
        :param in_channel: 输入特征维度
        :param mlp: list，表示最后接上的 MLP 各层维度
        '''
        super().__init__()

        self.n_near = n_near
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:  # mlp：数组
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.attention = AttentionInPnts(last_channel)

    def forward(self, xyz, fea):
        """
        Input:
            xyz: 点的 xyz 特征。input points position data, [B, C, N]
            points: 点的 ijk 特征。input points data, [B, D, N]
        Return:
            new_xyz: 处理后的 xyz 特征。sampled points position data, [B, C, S]
            new_points_concat: 处理后的 ijk 特征。sample points feature data, [B, D', S]
        """
        fea_neighbor = sample_and_group(self.n_near, xyz, fea)

        fea_neighbor = fea_neighbor.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        fea_center = torch.cat([torch.zeros_like(xyz, dtype=torch.float), fea], dim=-1)  # -> [B, npoint, C+D]
        fea_center = fea_center.unsqueeze(2).permute(0, 3, 2, 1)  # [B, C+D, 1, npoint]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            fea_neighbor = F.relu(bn(conv(fea_neighbor)))  # -> [B, emb, nsample, npoint]
            fea_center = F.relu(bn(conv(fea_center)))

        fea_center = fea_center.permute(0, 3, 2, 1)  # -> [B, npoint, nsample, emb]
        fea_neighbor = fea_neighbor.permute(0, 3, 2, 1)  # -> [B, npoint, 1, emb]
        fea_new = self.attention(fea_center, fea_neighbor)

        # # 每个采样点的邻近点的特征维度取最大值
        # fea_neighbor = torch.max(fea_neighbor, 2)[0]
        # fea_neighbor = fea_neighbor.permute(0, 2, 1)

        return fea_new


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

    def forward(self, fea1, fea2):
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
        new_fea = torch.cat([fea1, fea2], dim=-1)
        new_fea = new_fea.permute(0, 2, 1)  # -> [bs, emb, n_point]

        # 使用MLP对每个点的特征单独进行处理
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_fea = F.relu(bn(conv(new_fea)))

        new_fea = new_fea.permute(0, 2, 1)  # -> [bs, n_point, emb]

        return new_fea


class TriFeaPred(nn.Module):
    '''
    最初的基于pointnet++去除全局信息获得的三属性预测模型
    '''
    def __init__(self, n_metatype, n_embout=256, n_neighbor=100):
        '''
        :param n_metatype: 基元类别总数
        :param n_embout: 参数化模板特征提取主干输出特征长度
        :param n_neighbor: 每个点的邻域内点数，包含该点本身
        :param n_stepk: surface_knn 找点时每步的点数
        '''
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

        # 逐行回归欧拉角的MLP，属于回归
        eula_div = (n_embout / 3) ** 0.25
        self.eula_angle = full_connected_conv1d([n_embout, int(n_embout/eula_div), int(n_embout/eula_div**2), int(n_embout/eula_div**3), 3])

        # 逐行回归是否属于边界点的MLP，属于分类
        near_div = (n_embout / 2) ** 0.25
        self.edge_nearby = full_connected_conv1d([n_embout, int(n_embout/near_div), int(n_embout/near_div**2), int(n_embout/near_div**3), 2])

        # 逐行回归属于何种基元类别的MLP，属于分类
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
