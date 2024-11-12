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


def vis_stl(file_name):

    # 创建一个 MeshSet 对象
    ms = pymeshlab.MeshSet()

    # 从文件中加载网格
    ms.load_new_mesh(file_name)

    # 获取网格
    mesh = ms.current_mesh()

    # 获取顶点和面
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()

    # 绘制网格
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 创建 Poly3DCollection
    poly3d = Poly3DCollection(vertices[faces], alpha=0.1, edgecolor='k')
    ax.add_collection3d(poly3d)

    # 设置轴限制
    scale = vertices.flatten('F')
    ax.auto_scale_xyz(scale, scale, scale)

    # 显示图形
    plt.show()


def plot_rectangular_prism(ax, origin, size):
    """
    绘制一个长方体。

    参数：
    ax (Axes3D): Matplotlib 3D 轴。
    origin (tuple): 长方体的原点 (x, y, z)。
    size (tuple): 长方体的尺寸 (dx, dy, dz)。
    """
    # 长方体的顶点
    x = [origin[0], origin[0] + size[0]]
    y = [origin[1], origin[1] + size[1]]
    z = [origin[2], origin[2] + size[2]]

    # 定义长方体的 12 条边
    vertices = [[x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[0], y[1], z[0]],
                [x[0], y[0], z[1]], [x[1], y[0], z[1]], [x[1], y[1], z[1]], [x[0], y[1], z[1]]]

    # 定义长方体的 6 个面
    faces = [[vertices[j] for j in [0, 1, 5, 4]],
             [vertices[j] for j in [7, 6, 2, 3]],
             [vertices[j] for j in [0, 3, 7, 4]],
             [vertices[j] for j in [1, 2, 6, 5]],
             [vertices[j] for j in [0, 1, 2, 3]],
             [vertices[j] for j in [4, 5, 6, 7]]]

    # 创建 Poly3DCollection 对象
    poly3d = Poly3DCollection(faces, alpha=0.1, edgecolor='k', facecolors=[1,1,1])

    # 添加到轴
    ax.add_collection3d(poly3d)


class full_connected_conv3d(nn.Module):
    def __init__(self, channels: list, bias: bool = True, drop_rate: float = 0.4):
        '''
        构建全连接层，输出层不接 BatchNormalization、ReLU、dropout、SoftMax、log_SoftMax
        :param channels: 输入层到输出层的维度，[in, hid1, hid2, ..., out]
        :param drop_rate: dropout 概率
        '''
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
        '''
        :param embeddings: [bs, fea_in, n_row, n_col]
        :return: [bs, fea_out, n_row, n_col]
        '''
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
        '''
        构建全连接层，输出层不接 BatchNormalization、ReLU、dropout、SoftMax、log_SoftMax
        :param channels: 输入层到输出层的维度，[in, hid1, hid2, ..., out]
        :param drop_rate: dropout 概率
        '''
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
        '''
        :param embeddings: [bs, fea_in, n_row, n_col]
        :return: [bs, fea_out, n_row, n_col]
        '''
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
        '''
        构建全连接层，输出层不接 BatchNormalization、ReLU、dropout、SoftMax、log_SoftMax
        :param channels: 输入层到输出层的维度，[in, hid1, hid2, ..., out]
        :param drop_rate: dropout 概率
        '''
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
        '''
        :param embeddings: [bs, fea_in, n_points]
        :return: [bs, fea_out, n_points]
        '''
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
        '''
        构建全连接层，输出层不接 BatchNormalization、ReLU、dropout、SoftMax、log_SoftMax
        :param channels: 输入层到输出层的维度，[in, hid1, hid2, ..., out]
        :param drop_rate: dropout 概率
        '''
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
        '''
        :param embeddings: [bs, fea_in, n_points]
        :return: [bs, fea_out, n_points]
        '''
        fea = embeddings
        for i in range(self.n_layers - 2):
            fc = self.linear_layers[i]
            bn = self.batch_normals[i]
            drop = self.drop_outs[i]

            fea = drop(F.relu(bn(fc(fea))))

        fea = self.outlayer(fea)

        return fea


class SA_Attention(nn.Module):
    """
        从 pointnet++ set abstraction 改造过来的类似功能层
    """
    def __init__(self, n_center, n_near, in_channel, mlp):
        '''
        :param npoint: 使用FPS查找的中心点数，因此该点数也是池化到的点数
        :param radius: 沿每个中心点进行 ball query 的半径
        :param nsample: 每个ball里的点数最大值，感觉就是查找这个数目的点，和半径无关
        :param in_channel: 输入特征维度
        :param mlp: list，表示最后接上的 MLP 各层维度
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

    def forward(self, xyz, points):
        """
        Input:
            xyz: 点的 xyz 特征。input points position data, [B, C, N]
            points: 点的 ijk 特征。input points data, [B, D, N]
        Return:
            new_xyz: 处理后的 xyz 特征。sampled points position data, [B, C, S]
            new_points_concat: 处理后的 ijk 特征。sample points feature data, [B, D', S]
        """
        # 交换 xyz 的 1,2 维度，交换之前：torch.Size([24, 3, 1024])
        xyz = xyz.permute(0, 2, 1)

        # 对法向量特征进行处理
        if points is not None:
            points = points.permute(0, 2, 1)

        new_xyz, new_points = SA_SampleAndGroup(self.n_center, self.n_near, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1).to(torch.float)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # 每个采样点的邻近点的特征维度取最大值
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class SetAbstraction(nn.Module):
    """
        从 pointnet++ set abstraction 改造过来的类似功能层
    """
    def __init__(self, n_center, n_near, in_channel, mlp):
        '''
        :param npoint: 使用FPS查找的中心点数，因此该点数也是池化到的点数
        :param radius: 沿每个中心点进行 ball query 的半径
        :param nsample: 每个ball里的点数最大值，感觉就是查找这个数目的点，和半径无关
        :param in_channel: 输入特征维度
        :param mlp: list，表示最后接上的 MLP 各层维度
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

    def forward(self, xyz, points):
        """
        Input:
            xyz: 点的 xyz 特征。input points position data, [B, C, N]
            points: 点的 ijk 特征。input points data, [B, D, N]
        Return:
            new_xyz: 处理后的 xyz 特征。sampled points position data, [B, C, S]
            new_points_concat: 处理后的 ijk 特征。sample points feature data, [B, D', S]
        """
        # 交换 xyz 的 1,2 维度，交换之前：torch.Size([24, 3, 1024])
        xyz = xyz.permute(0, 2, 1)

        # 对法向量特征进行处理
        if points is not None:
            points = points.permute(0, 2, 1)

        new_xyz, new_points = SA_SampleAndGroup(self.n_center, self.n_near, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1).to(torch.float)  # [B, C+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # 每个采样点的邻近点的特征维度取最大值
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class SA_Scratch(nn.Module):
    """
        从 pointnet++ set abstraction 改造过来的类似功能层
        pointnet++ set abstraction 使用向量拼接特征
        初始改造
    """
    def __init__(self, n_center, n_near, in_channel, mlp):
        '''
        :param n_center: 使用FPS查找的中心点数，因此该点数也是池化到的点数, 输入None时不进行下采样
        :param n_near: 沿每个中心点进行 ball query 的半径
        :param in_channel: 输入特征维度
        :param mlp: list，表示最后接上的 MLP 各层维度
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

    def forward(self, xyz, fea):
        """
        Input:
            xyz: 点的 xyz 特征。input points position data, [B, C, N]
            points: 点的 ijk 特征。input points data, [B, D, N]
        Return:
            new_xyz: 处理后的 xyz 特征。sampled points position data, [B, C, S]
            new_fea_concat: 处理后的 ijk 特征。sample points feature data, [B, D', S]
        """
        # 交换 xyz 的 1,2 维度，交换之前：torch.Size([24, 3, 1024])
        xyz = xyz.permute(0, 2, 1)

        # 对法向量特征进行处理
        if fea is not None:
            fea = fea.permute(0, 2, 1)

        new_xyz, new_fea = SA_SampleAndGroup(self.n_center, self.n_near, xyz, fea)

        new_fea = new_fea.permute(0, 3, 2, 1).to(torch.float)  # [B, C+D, n_near, n_center]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_fea = F.relu(bn(conv(new_fea)))

        # 每个采样点的邻近点的特征维度取最大值
        new_fea = torch.max(new_fea, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_fea


class SA_Attention_test3(nn.Module):
    """
        从 pointnet++ set abstraction 改造过来的类似功能层
        pointnet++ set abstraction 使用向量拼接特征
        在每个通道上进行注意力机制，原始版本使用 max
    """
    def __init__(self, n_center, n_near, dim_in, mlp, dim_qkv):
        '''
        :param n_center: 使用FPS查找的中心点数，因此该点数也是池化到的点数, 输入None时不进行下采样
        :param n_near: 沿每个中心点进行 ball query 的半径
        :param in_channel: 输入特征维度
        :param mlp: list，表示最后接上的 MLP 各层维度
        '''
        super().__init__()

        self.n_center = n_center
        self.n_near = n_near
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = dim_in

        for out_channel in mlp:  # mlp：数组
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
        """
        Input:
            xyz: 点的 xyz 特征。input points position data, [B, C, N]
            points: 点的 ijk 特征。input points data, [B, D, N]
        Return:
            new_xyz: 处理后的 xyz 特征。sampled points position data, [B, C, S]
            new_fea_concat: 处理后的 ijk 特征。sample points feature data, [B, D', S]
        """
        # # 交换 xyz 的 1,2 维度，交换之前：torch.Size([24, 3, 1024])
        # xyz = xyz.permute(0, 2, 1)
        #
        # # 对法向量特征进行处理
        # if fea is not None:
        #     fea = fea.permute(0, 2, 1)

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

        # 位置参数 delta
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
    """
        从 pointnet++ set abstraction 改造过来的类似功能层
        pointnet++ set abstraction 使用向量拼接特征
        这里使用注意力机制改造，传统标量注意力机制
    """
    def __init__(self, n_center, n_near, dim_in, dim_qk, dim_v, mlp: list):
        '''
        :param n_center: 使用FPS查找的中心点数，因此该点数也是池化到的点数, 输入None时不进行下采样
        :param n_near: 沿每个中心点进行 ball query 的半径
        :param in_channel: 输入特征维度
        :param mlp: list，表示最后接上的 MLP 各层维度
        '''
        super().__init__()

        self.n_center = n_center
        self.n_near = n_near

        self.matq = nn.Conv2d(dim_in, dim_qk, 1, bias=False)
        self.matk = nn.Conv2d(dim_in, dim_qk, 1, bias=False)
        self.matv = nn.Conv2d(dim_in, dim_v, 1, bias=False)

        self.fea_final = full_connected_conv1d(mlp)

    def forward(self, xyz, fea):
        """
        Input:
            xyz: 点的 xyz 特征。input points position data, [B, C, N]
            points: 点的 ijk 特征。input points data, [B, D, N]
        Return:
            new_xyz: 处理后的 xyz 特征。sampled points position data, [B, C, S]
            new_fea_concat: 处理后的 ijk 特征。sample points feature data, [B, D', S]
        """
        # -> xyz: [bs, n_points, 3]
        # -> fea: [bs, n_points, dim_in]

        idx_surfknn_all = surface_knn(xyz, self.n_near, 10)

        if self.n_center is None:
            center_xyz = xyz
            center_fea_orig = fea
            idx = idx_surfknn_all
            grouped_xyz = index_points(xyz, idx_surfknn_all)

        else:
            fps_idx = farthest_point_sample(xyz, self.n_center)  # 采样后的点索引 troch.size([B, npoint])
            center_xyz = index_points(xyz, fps_idx)  # 获取 xyz 中，索引 fps_idx 对应的点
            center_fea_orig = index_points(fea, fps_idx)
            idx = index_points(idx_surfknn_all, fps_idx)
            grouped_xyz = index_points(xyz, idx)  # [B, n_center, n_near, 3]

        # 使用注意力机制更新权重
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
    """
        从 pointnet++ set abstraction 改造过来的类似功能层
        pointnet++ set abstraction 使用向量拼接特征
        这里使用注意力机制改造，向量注意力机制
    """
    def __init__(self, n_center, n_near, dim_in, dim_qkv, mlp: list):
        '''
        :param n_center: 使用FPS查找的中心点数，因此该点数也是池化到的点数, 输入None时不进行下采样
        :param n_near: 沿每个中心点进行 ball query 的半径
        :param in_channel: 输入特征维度
        :param mlp: list，表示最后接上的 MLP 各层维度
        '''
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
        """
        Input:
            xyz: 点的 xyz 特征。input points position data, [B, C, N]
            points: 点的 ijk 特征。input points data, [B, D, N]
        Return:
            new_xyz: 处理后的 xyz 特征。sampled points position data, [B, C, S]
            new_fea_concat: 处理后的 ijk 特征。sample points feature data, [B, D', S]
        """
        # -> xyz: [bs, n_points, 3]
        # -> fea: [bs, n_points, emb_in]

        idx_surfknn_all = surface_knn(xyz, self.n_near, 10)

        if self.n_center is None:
            center_xyz = xyz
            center_fea_orig = fea
            idx = idx_surfknn_all
            grouped_xyz = index_points(xyz, idx_surfknn_all)

        else:
            fps_idx = farthest_point_sample(xyz, self.n_center)  # 采样后的点索引 troch.size([B, npoint])
            center_xyz = index_points(xyz, fps_idx)  # 获取 xyz 中，索引 fps_idx 对应的点
            center_fea_orig = index_points(fea, fps_idx)
            idx = index_points(idx_surfknn_all, fps_idx)
            grouped_xyz = index_points(xyz, idx)  # [B, n_center, n_near, 3]

        # 使用注意力机制更新权重
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

        # 位置参数 delta
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


def correct_rate(pred, target):
    '''
    计算预测准确率
    :param pred: [bs, n_classes]
    :param label: [bs, ]
    :return: correct_rate
    '''
    bs = pred.size()[0]
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.long().data).cpu().sum()
    return correct.item() / float(bs)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


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
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

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
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        # 使用MLP对每个点的特征单独进行处理
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points


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


def SA_SampleAndGroup(n_center, n_near, xyz, fea, is_backecenter=False):
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

    idx_surfknn_all = surface_knn(xyz, n_near, 10)

    if n_center is None:
        new_xyz = xyz
        idx = idx_surfknn_all
        grouped_xyz = index_points(xyz, idx_surfknn_all)

    else:
        fps_idx = farthest_point_sample(xyz, n_center)  # 采样后的点索引 troch.size([B, npoint])
        new_xyz = index_points(xyz, fps_idx)  # 获取 xyz 中，索引 fps_idx 对应的点
        idx = index_points(idx_surfknn_all, fps_idx)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]

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


class SampleAndGroup_AttentionLayer_deprecated(nn.Module):
    '''
    用于取代 pointnet++ 中 sample and group 函数
    '''
    def __init__(self, n_center, n_near, dim_in, dim_qk, dim_v):
        super().__init__()
        # self.n_center = n_center
        # self.n_near = n_near

        self.matq = nn.Linear(dim_in, dim_qk, bias=False)
        self.matk = nn.Linear(dim_in, dim_qk, bias=False)
        self.matv = nn.Linear(dim_in, dim_v, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim_qk),
            nn.ReLU(),
            nn.Linear(dim_qk, dim_v)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim_v, dim_v),
            nn.ReLU(),
            nn.Linear(dim_v, dim_v),
        )

    def forward(self, n_center, n_near, xyz, fea):

        B, N, C = xyz.shape

        idx_surfknn_all = surface_knn(xyz, n_near, 10)

        if n_center is None:
            new_xyz = xyz
            idx = idx_surfknn_all
            grouped_xyz = index_points(xyz, idx_surfknn_all)
            n_centerreal = N

        else:
            fps_idx = farthest_point_sample(xyz, n_center)  # 采样后的点索引 troch.size([B, npoint])
            new_xyz = index_points(xyz, fps_idx)  # 获取 xyz 中，索引 fps_idx 对应的点
            idx = index_points(idx_surfknn_all, fps_idx)
            grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
            n_centerreal = n_center

        # 使用注意力机制更新权重
        if fea is not None:
            grouped_points = index_points(fea, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
        else:
            new_points = grouped_xyz_norm







        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, n_centerreal, 1, C)

        if fea is not None:
            grouped_points = index_points(fea, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
        else:
            new_points = grouped_xyz_norm

        return new_xyz, new_points


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


def get_neighbor_index(vertices: "(bs, vertice_num, 3)",  neighbor_num: int, is_backdis: bool = False):
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


def index_points(points, idx, is_label: bool = False):
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

    if is_label:
        new_points = points[batch_indices, idx]
    else:
        new_points = points[batch_indices, idx, :]

    return new_points


def surface_knn__(ind_neighbor: "(bs, n_pnt, n_stepk)", ind_target: int, k_near: int):
    # 初始化结果Tensor，形状为(bs, k_near)
    # 注意，这里我们假设k_near不会大于任何点的邻近点总数
    bs, n_pnt, n_stepk = ind_neighbor.size
    assert k_near > n_stepk

    results = torch.zeros(bs, k_near, dtype=torch.long)

    # 用于跟踪已经找到的邻近点，以避免重复
    found_neighbors = torch.zeros(bs, n_pnt, dtype=torch.bool)

    # 递归函数，用于扩展邻近点
    def expand(i_bs):
        nonlocal num_found
        # 获取当前点的邻近点
        current_neighbors = ind_neighbor[i_bs, ind_target, :]

        # 遍历当前点的邻近点
        for neighbor in current_neighbors:
            # 如果这个邻近点还没有被找到过
            if not found_neighbors[i_bs, neighbor]:
                # 标记为已找到
                found_neighbors[i_bs, neighbor] = True
                # 如果达到k_near，停止扩展
                if num_found < k_near:
                    results[i_bs, num_found] = neighbor
                    num_found += 1
                    # 如果已经找到足够的邻近点，结束递归
                    if num_found == k_near:
                        return
                # 否则，继续递归扩展这个邻近点的邻近点
                expand(neighbor, ind_target + 1)

    # 从ind_target索引开始扩展邻近点
    for i in range(bs):
        # 初始化已找到的邻近点数量
        num_found = 0

        expand(i, ind_target)

    return results[:, :k_near]  # 截断结果Tensor以确保正确的形状

def surface_knn_(ind_neighbor: "(bs, n_pnt, n_stepk)", ind_target: int, k_near: int):
    """
    寻找最近的 k_near 个邻近点索引
    Args:
        neighbor_index (torch.Tensor): 邻近点索引张量，形状为 (bs, n_pnt, n_stepk)
        ind_target (torch.Tensor): 目标索引张量，形状为 (bs,)
        k_near (int): 最近邻居数量

    Returns:
        torch.Tensor: 最近的 k_near 个邻近点索引，形状为 (bs, k_near)
    """
    bs, n_pnt, n_stepk = ind_neighbor.shape
    result = torch.zeros((bs, k_near), dtype=torch.long)

    for i in range(bs):
        inds = torch.zeros(n_pnt, dtype=torch.bool)
        inds[ind_target] = True
        count = 0
        while count < k_near:
            neighbors = ind_neighbor[i, inds]
            neighbors = neighbors[~inds[neighbors]]
            if neighbors.numel() == 0:
                break
            inds[neighbors] = True
            count += neighbors.numel()
        result[i, :count] = inds.nonzero(as_tuple=False).squeeze(-1)[:k_near]

    return result

def surface_knn_all(points_all: "(bs, n_pnts, 3)", k_near: int = 100, n_stepk = 10):
    '''
    :param points_all: 所有点坐标
    :param ind_neighbor_all: 索引为i的行代表第i个点的 knn 索引
    :param k_near: 邻近点数
    :return: (bs, n_pnt, k_near): 索引为i的行代表第i个点的 surface_knn 索引
    '''
    # return get_neighbor_index(points_all, k_near)

    ind_neighbor_all = get_neighbor_index(points_all, n_stepk)

    bs, n_pnts, _ = points_all.size()

    surface_near_all = torch.zeros(bs, n_pnts, k_near, dtype=torch.int)
    for i in range(n_pnts):
        surface_near_all[:, i, :] = surface_knn_single(points_all, ind_neighbor_all, i, k_near - 1)

    return surface_near_all


def surface_knn_single(points_all: "(bs, n_pnt, 3)", ind_neighbor_all: "(bs, n_pnt, n_stepk)", ind_target: int, k_near: int = 100):
    '''
    :param points_all: 点云中的全部点
    :param ind_neighbor_all: (bs, n_pnt, n_stepk): 索引为i的行代表第i个点的 knn 索引
    :param ind_target: 目标点索引
    :param k_near: 目标点邻近点的数量，不包含该点本身
    :param n_stepk: 单次寻找过程中，每个点的邻近点数
    :return: (bs, k_near + 1)，即 ind_target 对应的点的 k_near 个近邻，加上 k_near 自己，且自己处于第一个位置
    '''
    # 注意，这里我们假设k_near不会大于任何点的邻近点总数
    bs, n_pnt, _ = ind_neighbor_all.size()

    # 初始化, 创建全为 -1 的整形数组
    results = torch.zeros(bs, k_near + 1, dtype=torch.int)

    # 递归函数，用于扩展邻近点
    def expand():
        nonlocal current_neighbors

        # 递归终止条件
        num_current_neighbor = len(current_neighbors)
        overall_pnts = int(k_near * 1.2)
        # if len(current_neighbors) >= int(k_near * 1.5):
        if num_current_neighbor >= overall_pnts:
            return

        freeze_neighbors = current_neighbors.copy()

        # 遍历当前点的邻近点
        for neighbor in freeze_neighbors:
            # 找到该邻近点的所有其它邻近点索引
            sub_neighbors = [item.item() for item in currentbs_all_neighbors[neighbor, :]]

            for new_neighbor in sub_neighbors:
                # 如果这个邻近点还没有被找到过
                if new_neighbor not in current_neighbors:
                    current_neighbors.append(new_neighbor)

        expand()

    # 从ind_target索引开始扩展邻近点
    for i in range(bs):
        # 已有的索引放入一个列表, 后续向其中补充值
        current_neighbors = [item.item() for item in ind_neighbor_all[i, ind_target, :]]
        current_neighbors.append(ind_target)

        currentbs_all_neighbors = ind_neighbor_all[i, :, :]

        expand()

        # 从 current_neighbors 中找到与 ind_target 最近的 k_near 个点，使用红黑树字典实现升序排序
        wbtree = SortedDict()

        for near_neighbor in current_neighbors:
            wbtree[torch.dist(points_all[i, ind_target, :], points_all[i, near_neighbor, :], p=2).item()] = near_neighbor

        current_neighbors.clear()
        # 从该红黑树取 k_near + 1 个数
        items_count = 0

        for _, near_neighbor in wbtree.items():
            current_neighbors.append(near_neighbor)

            items_count += 1
            if items_count == k_near + 1:
                break

        assert len(current_neighbors) == k_near + 1
        results[i, :] = torch.tensor(current_neighbors)

    return results


def show_knn(points, center_ind: int, n_neighbor: int, bs) -> None:

    highlight_indices = get_neighbor_index(points, n_neighbor)

    # 生成示例整型数组，表示指定索引的点
    highlight_indices = highlight_indices[bs, center_ind, :]  # 示例指定的索引
    points = points[bs, :, :]

    # 绘制点云
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    highlight_points = points[highlight_indices]

    center_pnts = points[center_ind, :]

    new_element = torch.tensor([center_ind])
    delete_inds = torch.cat((highlight_indices, new_element))
    points = np.delete(points, delete_inds, axis=0)

    # 绘制所有点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=[110/255, 189/255, 183/255], label='Other Points', s=5)

    # 绘制指定索引的点，并设置为不同的颜色

    ax.scatter(highlight_points[:, 0], highlight_points[:, 1], highlight_points[:, 2], c='r',
               label='Highlighted Points', alpha=1, s=15)

    ax.scatter(center_pnts[0], center_pnts[1], center_pnts[2], c='b',
               label='Highlighted Points', alpha=1, s=25)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 添加图例
    ax.legend()
    ax.set_aspect('equal', adjustable='box')


    ## 画长方体
    # 定义长方体的原点和尺寸
    origin = (0, 0, 0)
    size = (150, 75, 10)

    # 绘制长方体
    plot_rectangular_prism(ax, origin, size)

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_axis_off()
    ax.view_init(elev=23, azim=45)

    plt.show()


def test():

    def read_coor(file_name, n_points):
        points_all = np.loadtxt(file_name)
        # choice = np.random.choice(points_all.shape[0], n_points, replace=True)
        # points_all = points_all[choice, :]

        return torch.from_numpy(points_all)

    def generate_batch(*files, n_points):
        all_coor = []

        for curr_file in files:
            all_coor.append(read_coor(curr_file, n_points).unsqueeze(0))

        all_coor = torch.cat(all_coor, dim=0)

        return all_coor

    # pointind = 865
    # pointind = 456
    # pointind = 782
    pointind = 79
    num_nei = 100
    n_stepk = 10
    n_points = 3058

    file_path0 = r'D:\document\DeepLearning\ParPartsNetWork\data_set\cuboid\PointCloud0.txt'
    file_path1 = r'D:\document\DeepLearning\ParPartsNetWork\data_set\cuboid\PointCloud1.txt'
    file_path2 = r'C:\Users\ChengXi\Desktop\hardreads\cuboid.txt'

    points = generate_batch(file_path2, n_points=n_points)

    for bs in range(points.size()[0]):

        show_knn(points, pointind, num_nei, bs)

        show_surfknn(points, bs, pointind, num_nei, n_stepk)

    exit(0)



    # 获取单步knn索引
    ind_neighbor_all = get_neighbor_index(points, n_stepk)

    start_time = time.time()
    surf_knn_all = surface_knn(points, num_nei, n_stepk)
    end_time = time.time()

    print('新SurfaceKNN时间消耗：', end_time - start_time)

    start_time = time.time()
    # surf_knn_all = surface_knn_all(points, num_nei, n_stepk)
    end_time = time.time()

    print('旧SurfaceKNN时间消耗：', end_time - start_time)

    new_near = surf_knn_all[:, pointind, :]

    # 获取SurfaceKNN近邻索引
    # new_near = surface_knn(points, ind_neighbor_all, pointind, num_nei)

    # 将索引转化为点坐标
    new_points = index_points(points, new_near)

    # 取第0批量的点作为显示
    points_show = new_points[0, :, :]

    # 找到中心点，高亮显示
    center_pnts = points[0, pointind, :]

    # 取第0批量所有点显示完整点云
    points_show_all = points[0, :, :]

    # 删除重复显示的点
    points_show_all = np.delete(points_show_all, new_near[0, :], axis=0)
    points_show = np.delete(points_show, 0, axis=0)

    # 设置matplotlib参数
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 显示所有点
    ax.scatter(points_show_all[:, 0], points_show_all[:, 1], points_show_all[:, 2], c='g', label='Other Points', alpha=0.8, s=5)

    # 显示邻近点
    ax.scatter(points_show[:, 0], points_show[:, 1], points_show[:, 2], c='r', label='Near Points', s=15)

    # 显示中心点
    ax.scatter(center_pnts[0], center_pnts[1], center_pnts[2], c='b', label='Center Points', s=25)

    # 设置matplotlib参数
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 添加图例
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    plt.show()


    # # 生成示例整型数组，表示指定索引的点
    # highlight_indices = new_near[0, :]
    # points = points[0, :, :]
    # highlight_points = points[highlight_indices, :]
    # center_pnts = points[pointind, :]
    #
    # points = np.delete(points, highlight_indices, axis=0)
    #
    # # 绘制点云
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 绘制所有点
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='g', label='Other Points', alpha=0.8, s=5)
    #
    # # 绘制指定索引的点，并设置为不同的颜色
    #
    # ax.scatter(highlight_points[:, 0], highlight_points[:, 1], highlight_points[:, 2], c='r',
    #            label='Highlighted Points', s=15)
    #
    # ax.scatter(center_pnts[0], center_pnts[1], center_pnts[2], c='b',
    #            label='Highlighted Points', s=25)
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    #
    # # 添加图例
    # ax.legend()
    # ax.set_aspect('equal', adjustable='box')
    #
    # plt.show()
    #
    # # results = torch.full((12,), -1, dtype=torch.int)
    # # curr_target_neighbor = [item.item() for item in results]
    # # print(curr_target_neighbor)


def patch_interpolate():
    # 生成随机曲面上的点数据
    x = np.random.rand(100)
    y = np.random.rand(100)
    z = np.sin(x * 2 * np.pi) * np.cos(y * 2 * np.pi)

    # 生成随机点的坐标
    points = np.column_stack((x, y))

    # 定义网格，用于生成拟合曲面
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))

    # 使用 griddata 函数拟合曲面
    grid_z = griddata(points, z, (grid_x, grid_y), method='cubic')

    # 绘制原始数据点和拟合曲面
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o', label='Original Points')
    ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def indexes_val(vals, inds):
    '''
    将索引替换为对应的值
    :param vals: size([bs, n_item, n_channel])
    :param inds: size([bs, n_item, n_vals])
    :return: size([bs, n_item, n_vals])
    '''
    bs, n_item, n_vals = inds.size()

    # 生成0维度索引
    sequence = torch.arange(bs)
    sequence_expanded = sequence.unsqueeze(1)
    sequence_3d = sequence_expanded.tile((1, n_item))
    sequence_4d = sequence_3d.unsqueeze(-1)
    batch_indices = sequence_4d.repeat(1, 1, n_vals)

    # 生成1维度索引
    view_shape = [n_item, n_vals]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = [bs, n_item, n_vals]
    repeat_shape[1] = 1
    channel_indices = torch.arange(n_item, dtype=torch.long).view(view_shape).repeat(repeat_shape)

    return vals[batch_indices, channel_indices, inds]


def surface_knn(points_all: "(bs, n_pnts, 3)", k_near: int = 100, n_stepk = 10):
    '''
    邻近点不包含自身
    :param points_all: 所有点坐标
    :param ind_neighbor_all: 索引为i的行代表第i个点的 knn 索引
    :param k_near: 邻近点数
    :return: (bs, n_pnt, k_near): 索引为i的行代表第i个点的 surface_knn 索引
    '''
    # 获取所有附近点的索引
    ind_neighbor_all, all_dist = get_neighbor_index(points_all, n_stepk, True)

    # 找到每行距离最大的索引
    neighbor_index_max = torch.max(all_dist, dim=-1, keepdim=True)[1]

    new_neighinds = ind_neighbor_all.clone()

    num_ita = 0
    while True:
        n_current_neighbors = new_neighinds.size()[-1]
        indexed_all = []
        for j in range(n_current_neighbors):
            indexed_all.append(index_points(ind_neighbor_all, new_neighinds[:, :, j]))
        new_neighinds = torch.cat(indexed_all, dim=-1)

        ## 去掉每行重复数
        # 先将相同数进行聚集，默认升序排列
        new_neighinds = torch.sort(new_neighinds, dim=-1)[0]

        # 将重复的第二个起替换成距离最大的索引
        duplicates = torch.zeros_like(new_neighinds)
        duplicates[:, :, 1:] = new_neighinds[:, :, 1:] == new_neighinds[:, :, :-1]

        neighbor_index_max2 = neighbor_index_max.repeat(1, 1, new_neighinds.shape[-1])
        new_neighinds[duplicates.bool()] = neighbor_index_max2[duplicates.bool()]

        ## 找到每行有效项目数
        # 将索引转换成距离数值
        dist_neighinds = indexes_val(all_dist, new_neighinds)

        # 将距离数值升序排序，最大的即为无效
        sort_dist = torch.sort(dist_neighinds, dim=-1)[0]  # -> [bs, n_point, n_near]

        # 找到最大的位置索引
        sort_dist_maxind = torch.max(sort_dist, dim=-1)[1]  # -> [bs, n_point]
        valid_nnear = torch.min(sort_dist_maxind).item() + 1

        is_end_loop = False
        if valid_nnear >= k_near + 1:
            valid_nnear = k_near + 1
            is_end_loop = True

        ## 找到距离最小的数的前k个数的索引
        sub_neighbor_index = torch.topk(dist_neighinds, k=valid_nnear, dim=-1, largest=False)[1]  # [0] val, [1] index

        # 然后将索引转化为对应点索引
        new_neighinds = indexes_val(new_neighinds, sub_neighbor_index)

        # 去掉自身
        new_neighinds = new_neighinds[:, :, 1:]

        if is_end_loop:
            break

        num_ita += 1
        if num_ita > 20:
            print('surface knn中达最大迭代次数，返回普通knn结果')
            return ind_neighbor_all

    return new_neighinds


def surface_knn_all_test_v2(points_all: "(bs, n_pnts, 3)", k_near: int = 100, n_stepk = 10):
    '''
    :param points_all: 所有点坐标
    :param ind_neighbor_all: 索引为i的行代表第i个点的 knn 索引
    :param k_near: 邻近点数
    :return: (bs, n_pnt, k_near): 索引为i的行代表第i个点的 surface_knn 索引
    '''
    # return get_neighbor_index(points_all, k_near)

    # 获取所有附近点的索引
    ind_neighbor_all, all_dist = get_neighbor_index(points_all, n_stepk, True)

    # 找到每行距离最大的索引
    neighbor_index_max = torch.topk(all_dist, k=1, dim=-1, largest=True)[1]

    # n_iters = 3
    n_iters = math.ceil(k_near / n_stepk) - 1

    new_neighinds = ind_neighbor_all.clone()

    n_current_near = n_stepk
    for i in range(k_near - n_stepk):
        n_current_near += 1

        n_current_neighbors = new_neighinds.size()[-1]
        indexed_all = []
        for j in range(n_current_neighbors):
            indexed_all.append(index_points(ind_neighbor_all, new_neighinds[:, :, j]))
        new_neighinds = torch.cat(indexed_all, dim=-1)

        ## 去掉每行重复数
        # 先将相同数进行聚集，默认升序排列
        new_neighinds = torch.sort(new_neighinds, dim=-1)[0]

        # 将重复的第二个起替换成距离最大的索引
        duplicates = torch.zeros_like(new_neighinds)
        duplicates[:, :, 1:] = new_neighinds[:, :, 1:] == new_neighinds[:, :, :-1]

        neighbor_index_max2 = neighbor_index_max.repeat(1, 1, new_neighinds.shape[-1])
        new_neighinds[duplicates.bool()] = neighbor_index_max2[duplicates.bool()]

        # 将索引转换成距离数值
        dist_neighinds = indexes_val(all_dist, new_neighinds)

        # 然后找出对应的最小的 n_current_near 个点
        sub_neighbor_index = torch.topk(dist_neighinds, k=n_current_near, dim=-1, largest=False)[1]  # [0] val, [1] index

        # 然后将索引转化为对应点索引
        new_neighinds = indexes_val(new_neighinds, sub_neighbor_index)

    return new_neighinds


def surface_knn_all_test(points_all: "(bs, n_pnts, 3)", k_near: int = 100, n_stepk = 10):
    '''
    :param points_all: 所有点坐标
    :param ind_neighbor_all: 索引为i的行代表第i个点的 knn 索引
    :param k_near: 邻近点数
    :return: (bs, n_pnt, k_near): 索引为i的行代表第i个点的 surface_knn 索引
    '''
    # return get_neighbor_index(points_all, k_near)

    # 获取所有附近点的索引
    ind_neighbor_all, all_dist = get_neighbor_index(points_all, n_stepk, True)

    # n_iters = 3
    n_iters = int(np.log(k_near) / np.log(n_stepk)) + 1

    new_neighinds = ind_neighbor_all.clone()
    for i in range(n_iters):
        n_current_neighbors = new_neighinds.size()[-1]
        indexed_all = []
        for j in range(n_current_neighbors):
            indexed_all.append(index_points(ind_neighbor_all, new_neighinds[:, :, j]))
        new_neighinds = torch.cat(indexed_all, dim=-1)

    # 找到每行距离最大的索引
    neighbor_index_max = torch.topk(all_dist, k=1, dim=-1, largest=True)[1]
    neighbor_index_max = neighbor_index_max.repeat(1, 1, new_neighinds.shape[-1])

    # 将重复的第二个起替换成最大的位置的索引
    new_neighinds = torch.unique(new_neighinds, dim=-1)

    duplicates = torch.zeros_like(new_neighinds)
    duplicates[:, :, 1:] = new_neighinds[:, :, 1:] == new_neighinds[:, :, :-1]
    new_neighinds[duplicates.bool()] = neighbor_index_max[duplicates.bool()]


    # 从中找到最近的一些点

    # 第一步，将索引替换成对应的值
    dist_neighinds = indexes_val(all_dist, new_neighinds)

    # 然后找出对应的最小的k_near个点
    sub_neighbor_index = torch.topk(dist_neighinds, k=k_near, dim=-1, largest=False)[1]  # [0] val, [1] index

    # 然后将索引转化为对应点索引
    final_neighinds = indexes_val(new_neighinds, sub_neighbor_index)

    return final_neighinds


def test_unique():
    # testtensor = torch.tensor([[1,2,3,4,4], [1,2,3,3,3]])
    # print(testtensor)
    # testtensor = torch.unique(testtensor, dim=-1, sorted=False)
    # print(testtensor)



    # 原始二维张量
    tensor = torch.tensor([[[1, 1, 3, 3], [1, 2, 2, 2]], [[1, 3, 3, 3], [1, 2, 3, 2]]])
    print(tensor)

    # 找到重复元素的位置
    duplicates = torch.zeros_like(tensor)
    duplicates[:, :, 1:] = tensor[:, :, 1:] == tensor[:, :, :-1]
    print(duplicates.bool())

    btensor = torch.tensor([[[0,  0, 0,  0],
         [1, 1,  1,  1]],

        [[0, 0,  0,  0],
         [1, 1, 1, 1]]])

    # 将重复元素替换为 999
    tensor[duplicates.bool()] = btensor[duplicates.bool()]

    print(tensor)


def test_knn2():
    # test_tensor = torch.rand(2, 5, 5)
    #
    # ind_tensor = torch.tensor([[[1, 2],
    #                            [2, 4],
    #                            [0, 3],
    #                            [1, 3],
    #                            [3, 4]],
    #
    #                            [[0, 1],
    #                             [3, 2],
    #                             [1, 2],
    #                             [3, 0],
    #                             [2, 3]]
    #                            ])
    #
    # print('test_tensor', test_tensor)
    # print('ind_tensor', ind_tensor)
    # print(indexes_val(test_tensor, ind_tensor))


    # bs = 2
    # n_item = 3
    # n_vals = 4
    #
    # view_shape = [n_item, n_vals]
    # view_shape[1:] = [1] * (len(view_shape) - 1)
    # repeat_shape = [bs, n_item, n_vals]
    # repeat_shape[1] = 1
    # batch_indices = torch.arange(n_item, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    #
    # print(batch_indices)




    test_tensor = torch.rand(2, 10, 3)
    print(test_tensor)
    # ind_tensor = torch.tensor([[1, 4, 6],
    #                            [7, 2, 9]]
    #                           )
    #
    # print('pnts: ', test_tensor)
    # print('inds: ', ind_tensor)
    #
    # indpnts = index_points(test_tensor, ind_tensor)
    #
    # print('res: ', indpnts)

    print(surface_knn_all_test(test_tensor, 5, 2))


def test_batch_indexes():
    test_data = torch.rand(2, 3, 4)

    test_ind = torch.tensor([[[1, 2],
                              [1, 2],
                              [1, 2]],

                             [[2, 3],
                              [2, 3],
                              [2, 3]]
                             ])

    print(test_data)
    print(test_ind)

    res = indexes_val(test_data, test_ind)

    print(res)
    exit(0)


    bs, n_item, n_vals = 2, 4, 3

    # 生成0维度索引
    sequence = torch.arange(bs)
    print(sequence)
    sequence_expanded = sequence.unsqueeze(1)
    print(sequence_expanded)
    sequence_3d = sequence_expanded.tile((1, n_item))
    print(sequence_3d)
    sequence_4d = sequence_3d.unsqueeze(-1)
    print(sequence_4d)
    batch_indices = sequence_4d.repeat(1, 1, n_vals)
    print(batch_indices)


def test_surfknn_testv2():
    testtensor = torch.rand(2, 10, 3)

    surface_knn_all_test_v2(testtensor, 8, 3)


def test_where():
    x = torch.tensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
    y = torch.tensor([[5, 6, 7], [7, 8, 9], [9, 10, 11]])
    z = torch.where(x > 5, x, y)


    print(f'x = {x}')
    print(f'=========================')
    print(f'y = {y}')
    print(f'=========================')
    print(f'x > 5 = {x > 5}')
    print(f'=========================')
    print(f'z = {z}')

    # > print
    # result:
    # x = tensor([[1, 2, 3],
    #             [3, 4, 5],
    #             [5, 6, 7]])
    # == == == == == == == == == == == == =
    # y = tensor([[5, 6, 7],
    #             [7, 8, 9],
    #             [9, 10, 11]])
    # == == == == == == == == == == == == =
    # x > 5 = tensor([[False, False, False],
    #                 [False, False, False],
    #                 [False, True, True]])
    # == == == == == == == == == == == == =
    # z = tensor([[5, 6, 7],
    #             [7, 8, 9],
    #             [9, 6, 7]])


def show_surfknn(points, bs=0, ind_center=865, n_neighbor=100, n_stepk=10):
    start_time = time.time()
    surf_knn_all = surface_knn(points, n_neighbor, n_stepk).cpu()
    end_time = time.time()
    points = points.cpu()

    print('新SurfaceKNN时间消耗：', end_time - start_time)

    new_near = surf_knn_all[:, ind_center, :]

    # 将索引转化为点坐标
    new_points = index_points(points, new_near)

    # 取第bs批量的点作为显示
    points_show = new_points[bs, :, :]

    # 找到中心点，高亮显示
    center_pnts = points[bs, ind_center, :]

    # 取第0批量所有点显示完整点云
    points_show_all = points[bs, :, :]

    points_show_all = points_show_all
    # 删除重复显示的点

    points_show_all = np.delete(points_show_all, new_near[bs, :], axis=0)
    points_show = np.delete(points_show, 0, axis=0)

    # 设置matplotlib参数
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 显示所有点
    ax.scatter(points_show_all[:, 0], points_show_all[:, 1], points_show_all[:, 2], c=[110/255, 189/255, 183/255], label='Other Points', s=5)

    # 显示邻近点
    ax.scatter(points_show[:, 0], points_show[:, 1], points_show[:, 2], c='r', label='Near Points', alpha=1, s=15)

    # 显示中心点
    ax.scatter(center_pnts[0], center_pnts[1], center_pnts[2], c='b', alpha=1, label='Center Points', s=25)

    # 设置matplotlib参数
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 添加图例
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    ## 画长方体
    # 定义长方体的原点和尺寸
    origin = (0, 0, 0)
    size = (150, 75, 10)

    # 绘制长方体
    plot_rectangular_prism(ax, origin, size)

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_axis_off()
    ax.view_init(elev=23, azim=45)

    # 显示图形
    plt.show()


def show_neighbor_points(points_all, points_neighbor, point_center) -> None:
    """
    显示所有点、中心点、邻近点，以测试 knn 是否正确
    :param points_all: [n_points, 3]
    :param points_neighbor: [n_points, n_neighbor]
    :param point_center: int
    :return: None
    """
    # 设置matplotlib参数
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 显示中心点
    ax.scatter(point_center[0], point_center[1], point_center[2], color='b', alpha=1, label='Center Points', s=25)

    # 显示邻近点
    ax.scatter(points_neighbor[:, 0], points_neighbor[:, 1], points_neighbor[:, 2], color='r', label='Near Points', alpha=1, s=15)

    # 显示所有点
    ax.scatter(points_all[:, 0], points_all[:, 1], points_all[:, 2], color=[110/255, 189/255, 183/255], label='Other Points', s=5)

    # 设置matplotlib参数
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 添加图例
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    ## 画长方体
    # 定义长方体的原点和尺寸
    origin = (0, 0, 0)
    size = (150, 75, 10)
    # origin = (-75, -75, -10)
    # size = (150, 75, 10)


    # 绘制长方体
    plot_rectangular_prism(ax, origin, size)

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_axis_off()
    ax.view_init(elev=23, azim=45)

    # 显示图形
    plt.show()


def surf_knn_pral():
    """
    用于显示 surface knn 的原理
    :return:
    """
    file_path = r'C:\Users\ChengXi\Desktop\hardreads\cuboid_view.txt'
    # center_ind = 512
    # center_ind = 200
    center_ind = 800
    n_neighbor = 90
    n_stepk = 10
    sample_func = surface_knn

    points_all = np.loadtxt(file_path, dtype=float)
    points_all = torch.from_numpy(points_all)
    points_all = points_all.unsqueeze(0).repeat(2, 1, 1)

    ind_surf_knn = sample_func(points_all, n_neighbor, n_stepk)

    neighbor_ind = ind_surf_knn[:, center_ind, :]

    # 将索引转化为点坐标
    neighbor_points = index_points(points_all, neighbor_ind)

    # 取第bs批量的点作为显示
    neighbor_points = neighbor_points[0, :, :]

    # 找到中心点，高亮显示
    center_pnt = points_all[0, center_ind, :]

    # 取第0批量所有点显示完整点云
    points_all = points_all[0, :, :]

    # 删除重复显示的点
    center_and_neighbor_ind = torch.cat([neighbor_ind[0, :], torch.tensor([center_ind])])
    points_all = np.delete(points_all, center_and_neighbor_ind, axis=0)

    np.savetxt('3.txt', center_and_neighbor_ind.numpy())

    show_neighbor_points(points_all, neighbor_points, center_pnt)


def show_surfknn_paper1():
    file_path = r'C:\Users\ChengXi\Desktop\hardreads\cuboid_view.txt'
    # center_ind = 512
    center_ind = 800

    points_all = np.loadtxt(file_path, dtype=float)
    center_point = points_all[center_ind, :]

    neighbor_ind1 = np.loadtxt('1.txt', dtype=int)[:-1]
    neighbor_points1 = points_all[neighbor_ind1, :]

    # 删除原有的重复显示的点
    neighbor_and_center_ind = torch.cat([torch.from_numpy(neighbor_ind1), torch.tensor([center_ind])])
    points_all = np.delete(points_all, neighbor_and_center_ind, axis=0)

    show_neighbor_points(points_all, neighbor_points1, center_point)


def show_surfknn_paper2():
    file_path = r'C:\Users\ChengXi\Desktop\hardreads\cuboid_view.txt'
    # center_ind = 512
    center_ind = 800

    points_all = np.loadtxt(file_path, dtype=float)

    # 第二幅图中，已存在点为第一步的中心点
    exist_point = points_all[center_ind, :]

    center_ind = np.loadtxt('1.txt', dtype=int)
    center_points = points_all[center_ind[:-1], :]

    neighbor_ind = np.loadtxt('2.txt', dtype=int)
    neighbor_ind = array_subtraction(neighbor_ind, center_ind)
    neighbor_points = points_all[neighbor_ind, :]

    # 删除原有的重复显示的点
    neighbor_and_center_ind = np.loadtxt('2.txt', dtype=int)
    points_all = np.delete(points_all, neighbor_and_center_ind, axis=0)


    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 显示已有点
    # [156/255, 64/255, 132/255]
    ax.scatter(exist_point[0], exist_point[1], exist_point[2], color='g', alpha=1, label='Center Points', s=25)

    # 显示中心点
    ax.scatter(center_points[:, 0], center_points[:, 1], center_points[:, 2], color='b', alpha=1, label='Center Points', s=25)

    # 显示邻近点
    ax.scatter(neighbor_points[:, 0], neighbor_points[:, 1], neighbor_points[:, 2], color='r', label='Near Points', alpha=1, s=15)

    # 显示所有点
    ax.scatter(points_all[:, 0], points_all[:, 1], points_all[:, 2], color=[110/255, 189/255, 183/255], label='Other Points', s=5)

    # 设置matplotlib参数
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 添加图例
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    ## 画长方体
    # 定义长方体的原点和尺寸
    origin = (0, 0, 0)
    size = (150, 75, 10)

    # 绘制长方体
    plot_rectangular_prism(ax, origin, size)

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_axis_off()
    ax.view_init(elev=23, azim=45)

    # 显示图形
    plt.show()


def show_surfknn_paper3():
    file_path = r'C:\Users\ChengXi\Desktop\hardreads\cuboid_view.txt'

    points_all = np.loadtxt(file_path, dtype=float)

    # 第二幅图中，已存在点为第一步的中心点
    exist_ind = np.loadtxt('1.txt', dtype=int)
    exist_point = points_all[exist_ind, :]

    center_ind = np.loadtxt('2.txt', dtype=int)
    center_ind2 = array_subtraction(center_ind, exist_ind)
    center_points = points_all[center_ind2[:-1], :]

    neighbor_ind = np.loadtxt('3.txt', dtype=int)
    neighbor_ind = array_subtraction(neighbor_ind, center_ind)
    neighbor_points = points_all[neighbor_ind, :]

    # 删除原有的重复显示的点
    neighbor_and_center_ind = np.loadtxt('3.txt', dtype=int)
    points_all = np.delete(points_all, neighbor_and_center_ind, axis=0)


    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 显示已有点
    # 156/255, 64/255, 132/255
    ax.scatter(exist_point[:, 0], exist_point[:, 1], exist_point[:, 2], color='g', alpha=1, label='Center Points', s=25)

    # 显示中心点
    ax.scatter(center_points[:, 0], center_points[:, 1], center_points[:, 2], color='b', alpha=1, label='Center Points', s=25)

    # 显示邻近点
    ax.scatter(neighbor_points[:, 0], neighbor_points[:, 1], neighbor_points[:, 2], color='r', label='Near Points', alpha=1, s=15)

    # 显示所有点
    ax.scatter(points_all[:, 0], points_all[:, 1], points_all[:, 2], color=[110/255, 189/255, 183/255], label='Other Points', s=5)

    # 设置matplotlib参数
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 添加图例
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    ## 画长方体
    # 定义长方体的原点和尺寸
    origin = (0, 0, 0)
    size = (150, 75, 10)

    # 绘制长方体
    plot_rectangular_prism(ax, origin, size)

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_axis_off()
    ax.view_init(elev=23, azim=45)

    # 显示图形
    plt.show()


def show_surfknn_paper4():
    file_path = r'C:\Users\ChengXi\Desktop\hardreads\cuboid_view.txt'

    points_all = np.loadtxt(file_path, dtype=float)

    # 第二幅图中，已存在点为第一步的中心点
    exist_ind = np.loadtxt('2.txt', dtype=int)
    exist_point = points_all[exist_ind, :]

    center_ind = np.loadtxt('3.txt', dtype=int)
    center_ind2 = array_subtraction(center_ind, exist_ind)
    center_points = points_all[center_ind2[:-1], :]

    neighbor_ind = np.loadtxt('4.txt', dtype=int)
    neighbor_ind = array_subtraction(neighbor_ind, center_ind)
    neighbor_points = points_all[neighbor_ind, :]

    # 删除原有的重复显示的点
    neighbor_and_center_ind = np.loadtxt('4.txt', dtype=int)
    points_all = np.delete(points_all, neighbor_and_center_ind, axis=0)


    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 显示已有点
    # 156/255, 64/255, 132/255
    ax.scatter(exist_point[:, 0], exist_point[:, 1], exist_point[:, 2], color='g', alpha=1, label='Center Points', s=25)

    # 显示中心点
    ax.scatter(center_points[:, 0], center_points[:, 1], center_points[:, 2], color='b', alpha=1, label='Center Points', s=25)

    # 显示邻近点
    ax.scatter(neighbor_points[:, 0], neighbor_points[:, 1], neighbor_points[:, 2], color='r', label='Near Points', alpha=1, s=15)

    # 显示所有点
    ax.scatter(points_all[:, 0], points_all[:, 1], points_all[:, 2], color=[110/255, 189/255, 183/255], label='Other Points', s=5)

    # 设置matplotlib参数
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 添加图例
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    ## 画长方体
    # 定义长方体的原点和尺寸
    origin = (0, 0, 0)
    size = (150, 75, 10)

    # 绘制长方体
    plot_rectangular_prism(ax, origin, size)

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_axis_off()
    ax.view_init(elev=23, azim=45)

    # 显示图形
    plt.show()


def teat_star():
    atensor = torch.tensor([[1, 2, 3],
                            [2, 3, 4],
                            [3, 4, 5],
                            [5, 3, 8]])

    btansor = torch.tensor([1, 2, 0, 4]).view(4, 1)

    print(btansor * atensor)


def array_subtraction(array_large, array_small):

    # 将数组转换为集合并进行差集运算
    set1 = set(array_small)
    set2 = set(array_large)
    result_set = set2 - set1

    # 将结果集合转换回Numpy数组
    result = np.array(list(result_set))

    return result


def show_different_weight_paper():

    def show_neighbor_diff_points(points_all, points_neighbor_valid, points_neighbor_invalid, point_center) -> None:
        # 设置matplotlib参数
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')

        # 显示中心点
        ax.scatter(point_center[0], point_center[1], point_center[2], color='b', alpha=1, label='Center Points', s=25)

        # 显示邻近点
        ax.scatter(points_neighbor_valid[:, 0], points_neighbor_valid[:, 1], points_neighbor_valid[:, 2], color='r', label='Near Points',
                   alpha=1, s=15)

        # 显示无效点
        ax.scatter(points_neighbor_invalid[:, 0], points_neighbor_invalid[:, 1], points_neighbor_invalid[:, 2], color=[79 / 255, 29 / 255, 97 / 255], label='Near Points',
                   alpha=1, s=15)

        # 显示所有点
        ax.scatter(points_all[:, 0], points_all[:, 1], points_all[:, 2], color=[110 / 255, 189 / 255, 183 / 255],
                   label='Other Points', s=5)

        # 设置matplotlib参数
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # 添加图例
        ax.legend()
        ax.set_aspect('equal', adjustable='box')

        ## 画长方体
        # 定义长方体的原点和尺寸
        # origin = (0, 0, 0)
        # size = (150, 75, 10)
        origin = (-75, -75, -10)
        size = (150, 75, 10)

        # 绘制长方体
        plot_rectangular_prism(ax, origin, size)

        # 设置轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_axis_off()
        ax.view_init(elev=23, azim=45)

        # 显示图形
        plt.show()

    file_path = 'face.txt'
    center_ind = 251
    n_stepk = 10
    n_neighbor = 100

    # 获取SurfaceKNN获得的邻近点
    points_all_raw = np.loadtxt(file_path, dtype=float)
    point_center = points_all_raw[center_ind, :]
    points_all = torch.from_numpy(points_all_raw).unsqueeze(0).repeat(2, 1, 1)
    neighbor_ind = surface_knn(points_all, n_neighbor, n_stepk)
    neighbor_ind = neighbor_ind[0, center_ind, :].numpy()

    # 周围点
    # points_around = points_all_raw[neighbor_ind, :]
    points_around1 = np.loadtxt('points1.txt', dtype=float)
    points_around2 = np.loadtxt('points2.txt', dtype=float)

    # 删除重复点
    neighbor_ind = np.append(neighbor_ind, center_ind)
    points_other = np.delete(points_all_raw, neighbor_ind, axis=0)

    # 显示点
    show_neighbor_diff_points(points_other, points_around1, points_around2, point_center)

    # 保存中心点
    # np.savetxt(r'C:\Users\ChengXi\Desktop\Drawing\array_data.txt', points_around)





if __name__ == '__main__':

    # 创建图形和 3D 轴
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 定义长方体的原点和尺寸
    # origin = (0, 0, 0)
    # size = (3, 2, 1)
    #
    # # 绘制长方体
    # plot_rectangular_prism(ax, origin, size)
    #
    # # 设置轴标签
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # # 设置轴范围
    # ax.set_xlim([0, 5])
    # ax.set_ylim([0, 5])
    # ax.set_zlim([0, 5])
    #
    # # 显示图形
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

    show_different_weight_paper()
