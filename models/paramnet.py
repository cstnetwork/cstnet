# 工具包
import torch.nn as nn
import torch
import torch.nn.functional as F

# 自定义模块
from utils import surface_knn_all
from utils import full_connected_conv1d, full_connected
from encoders import PN2CSEncoderParam
from encoders import pointnet2_cls_ssg_encoder, mini_encoder, pointnet2_partseg_encoder, MiniEncoder_PN2Seg, mini_encoder2
from pointnet2_utils import index_points
import encoders
import utils
from point_transformer_pytorch import PointTransformerLayer


class ParamNet_AddFeaturePredLayer(nn.Module):
    '''
    三项附加属性预测模型
    '''
    def __init__(self, n_metatype=4):
        super().__init__()

        emb_out = 256
        self.attention_l1 = encoders.AddAttr_AttentionLayer(process_channels=[3, 8, 16],
                                                            otherother_rate=0.2,
                                                            downsample_rate=0.8)

        self.attention_l2 = encoders.AddAttr_AttentionLayer(process_channels=[16, 32, 64],
                                                            otherother_rate=0.2,
                                                            downsample_rate=0.8)

        self.attention_l3 = encoders.AddAttr_AttentionLayer(process_channels=[64, 128, emb_out],
                                                            otherother_rate=0.2,
                                                            downsample_rate=0.8)

        # 逐行回归欧拉角的MLP，属于回归
        self.eula_angle = full_connected([emb_out, 128, 32, 3])

        # 逐行回归是否属于边界点的MLP，属于分类
        self.edge_nearby = full_connected([emb_out, 128, 32, 2])

        # 逐行回归属于何种基元类别的MLP，属于分类
        self.meta_type = full_connected([emb_out, 128, 64, n_metatype])

    def forward(self, point_center, points_nearby):
        '''
        :param point_center: [bs, channels_in]
        :param points_nearby: [bs, n_nearby, channels_in]
        :return:
        '''

        fea_center, fea_other = self.attention_l1(point_center, points_nearby)
        fea_center, fea_other = self.attention_l2(fea_center, fea_other)
        fea_center, _ = self.attention_l3(fea_center, fea_other)

        eula_angle = self.eula_angle(fea_center)
        # [bs, 3]

        edge_nearby = self.edge_nearby(fea_center)
        # [bs, 2]

        meta_type = self.meta_type(fea_center)
        # [bs, n_metatype]

        edge_nearby_log = F.log_softmax(edge_nearby, dim=-1)
        meta_type_log = F.log_softmax(meta_type, dim=-1)

        return eula_angle, edge_nearby_log, meta_type_log


class paramnet(nn.Module):
    def __init__(self, n_metatype=13, n_classes=2, n_embout=256, n_neighbor=100, n_stepk=10):
        '''
        :param n_metatype: 基元类别总数
        :param n_classes: 总类别数
        :param n_embout: 参数化模板特征提取主干输出特征长度
        :param n_neighbor: 每个点的邻域内点数，包含该点本身
        :param n_stepk: surface_knn 找点时每步的点数
        '''
        super().__init__()

        # 逐行回归欧拉角的MLP，属于回归
        self.eula_angle = MiniEncoder_PN2Seg(3)

        # 逐行回归是否属于边界点的MLP，属于分类
        self.edge_nearby = MiniEncoder_PN2Seg(2)

        # 逐行回归属于何种基元类别的MLP，属于分类
        self.meta_type = MiniEncoder_PN2Seg(n_metatype)

        # 拼接特征后进行分割的网络
        param_l1 = PN2CSEncoderParam()
        param_l1.append(npoint=512, radius=0.2, nsample=32, in_channel=3+3+2+n_metatype, mlp=[64, 64, 128], group_all=False)
        param_l1.append(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        param_l1.append(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)

        self.multiattr_classifier = pointnet2_cls_ssg_encoder(param_l1)
        self.final_type = full_connected([1024, 256, 64, n_classes])

    def forward(self, points_all):

        eula_angle = self.eula_angle(points_all)
        # [bs, n_points_all, 3]

        edge_nearby = self.edge_nearby(points_all)
        # [bs, n_points_all, 2]

        meta_type = self.meta_type(points_all)
        # [bs, n_points_all, 7]

        attr_all = torch.cat(
            (
                points_all,
                eula_angle,
                F.softmax(edge_nearby, dim=-1),
                F.softmax(meta_type, dim=-1)
            ), dim=-1)
        # [bs, n_points_all, 3 + 3 + 2 + 7]

        attr_all = self.multiattr_classifier(attr_all)

        attr_all = self.final_type(attr_all)
        final_type_log = F.log_softmax(attr_all, dim=-1)

        edge_nearby_log = F.log_softmax(edge_nearby, dim=-1)
        meta_type_log = F.log_softmax(meta_type, dim=-1)

        return final_type_log, eula_angle, edge_nearby_log, meta_type_log


class TriFeaSlim(nn.Module):
    '''
    最初的基于pointnet++去除全局信息获得的三属性预测模型
    '''
    def __init__(self, n_points_all, n_metatype=4, n_embout=128, n_neighbor=100, n_stepk=10):
        '''
        :param n_metatype: 基元类别总数
        :param n_embout: 参数化模板特征提取主干输出特征长度
        :param n_neighbor: 每个点的邻域内点数，包含该点本身
        :param n_stepk: surface_knn 找点时每步的点数
        '''
        super().__init__()

        self.n_neighbor = n_neighbor
        self.n_stepk = n_stepk

        rate_downsample = 0.9
        self.sa1 = utils.SetAbstraction(n_center=int(n_points_all * rate_downsample),
                                        n_near=100,
                                        in_channel=3 + 3,
                                        mlp=[8, 16, 32]
                                        )

        self.sa2 = utils.SetAbstraction(n_center=int(n_points_all * rate_downsample ** 2),
                                        n_near=50,
                                        in_channel=32 + 3,
                                        mlp=[64, 128, 256]
                                        )

        self.fp2 = utils.FeaPropagate(in_channel=256 + 32,
                                      mlp=[256, 128, 64]
                                      )  # in_chanell = points2_chanell + points1_channel

        self.fp1 = utils.FeaPropagate(in_channel=64 + 6,
                                      mlp=[128, 64, 32]
                                      )

        self.conv1 = nn.Conv1d(32, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, n_embout, 1)

        # 逐行回归欧拉角的MLP，属于回归
        self.eula_angle = full_connected_conv1d([n_embout, 128, 32, 3])

        # 逐行回归是否属于边界点的MLP，属于分类
        self.edge_nearby = full_connected_conv1d([n_embout, 128, 32, 2])

        # 逐行回归属于何种基元类别的MLP，属于分类
        self.meta_type = full_connected_conv1d([n_embout, 128, 64, n_metatype])

    def forward(self, xyz):
        # -> xyz: [bs, n_points, 3]
        xyz = xyz.transpose(1, -1)

        # Set Abstraction layers
        l0_points = xyz
        l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        # Feature Propagation layers
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)

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


class TriFeaValidV1(nn.Module):
    '''
    最初的基于pointnet++去除全局信息获得的三属性预测模型
    '''
    def __init__(self, n_points_all, n_metatype, n_embout=256, n_neighbor=100, n_stepk=10):
        '''
        :param n_metatype: 基元类别总数
        :param n_embout: 参数化模板特征提取主干输出特征长度
        :param n_neighbor: 每个点的邻域内点数，包含该点本身
        :param n_stepk: surface_knn 找点时每步的点数
        '''
        super().__init__()

        self.n_neighbor = n_neighbor
        self.n_stepk = n_stepk

        rate_downsample = 0.9
        self.sa1 = utils.SetAbstraction(n_center=int(n_points_all * rate_downsample), n_near=50, in_channel=3+3, mlp=[64, 64, 128])
        self.sa2 = utils.SetAbstraction(n_center=int(n_points_all * rate_downsample ** 2), n_near=100, in_channel=128+3, mlp=[128, 128, 256])

        self.fp2 = utils.FeaPropagate(in_channel=256+128, mlp=[256, 256, 128])  # in_chanell = points2_chanell + points1_channel
        self.fp1 = utils.FeaPropagate(in_channel=128+6, mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, n_embout, 1)

        # 逐行回归欧拉角的MLP，属于回归
        self.eula_angle = full_connected_conv1d([n_embout, 256, 128, 32, 3])

        # 逐行回归是否属于边界点的MLP，属于分类
        self.edge_nearby = full_connected_conv1d([n_embout, 256, 128, 32, 2])

        # 逐行回归属于何种基元类别的MLP，属于分类
        self.meta_type = full_connected_conv1d([n_embout, 256, 128, 64, n_metatype])

    def forward(self, xyz):
        # -> xyz: [bs, n_points, 3]

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
        # feat = self.conv1(l0_points)
        # feat = feat.permute(0, 2, 1)

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


class TriFeaPredAttention(nn.Module):
    '''
    基于注意力机制的三要素预测模型
    '''
    def __init__(self, n_points_all, n_metatype, n_embout=256, n_neighbor=100, n_stepk=10):
        '''
        :param n_metatype: 基元类别总数
        :param n_embout: 参数化模板特征提取主干输出特征长度
        :param n_neighbor: 每个点的邻域内点数，包含该点本身
        :param n_stepk: surface_knn 找点时每步的点数
        '''
        super().__init__()

        self.n_neighbor = n_neighbor
        self.n_stepk = n_stepk

        # pointnet++ encoder
        # self.backbone_trifea = encoders.PN2PartSegSsgEncoder_AttentionV1(channel_out=n_embout, n_points_all=n_points_all)
        self.backbone_trifea = encoders.PN2PartSegSsgEncoder_AttentionV2(channel_out=n_embout,
                                                                         n_points_all=n_points_all)
        # self.backbone_trifea = encoders.PN2PartSegSsgEncoder_AttentionV3(channel_out=n_embout,
        #                                                                  n_points_all=n_points_all)
        # self.backbone_trifea = encoders.PN2PartSegSsgEncoder_AttentionV4(channel_out=n_embout,
        #                                                                  n_points_all=n_points_all)
        # self.backbone_trifea = encoders.PN2PartSegSsgEncoder_AttentionV5(channel_out=n_embout,
        #                                                                  n_points_all=n_points_all)

        # 逐行回归欧拉角的MLP，属于回归
        self.eula_angle = full_connected_conv1d([n_embout, 256, 128, 32, 3])

        # 逐行回归是否属于边界点的MLP，属于分类
        self.edge_nearby = full_connected_conv1d([n_embout, 256, 128, 32, 2])

        # 逐行回归属于何种基元类别的MLP，属于分类
        self.meta_type = full_connected_conv1d([n_embout, 256, 128, 64, n_metatype])

    def forward(self, points_all):
        # -> points_all: [bs, n_points, 3]

        ex_features = self.backbone_trifea(points_all)
        # ex_features = self.backbone_trifea(all_assigned_pnts)
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


class TriFeaPred_SSG(nn.Module):
    def __init__(self, n_metatype=13, n_embout=256, n_neighbor=100, n_stepk=10):
        '''
        :param n_metatype: 基元类别总数
        :param n_embout: 参数化模板特征提取主干输出特征长度
        :param n_neighbor: 每个点的邻域内点数，包含该点本身
        :param n_stepk: surface_knn 找点时每步的点数
        '''
        super().__init__()

        self.n_neighbor = n_neighbor
        self.n_stepk = n_stepk

        # pointnet++ encoder
        self.backbone_trifea = encoders.PN2PartSegSsgEncoder()

        # 逐行回归欧拉角的MLP，属于回归
        self.eula_angle = full_connected_conv1d([n_embout, 256, 128, 32, 3])

        # 逐行回归是否属于边界点的MLP，属于分类
        self.edge_nearby = full_connected_conv1d([n_embout, 256, 128, 32, 2])

        # 逐行回归属于何种基元类别的MLP，属于分类
        self.meta_type = full_connected_conv1d([n_embout, 256, 128, 64, n_metatype])

    def forward(self, points_all):
        # -> points_all: [bs, n_points, 3]

        ex_features = self.backbone_trifea(points_all)
        # ex_features = self.backbone_trifea(all_assigned_pnts)
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


class TriFeaPred_Orig(nn.Module):
    '''
    最初版本可用的三属性预测模型
    '''
    def __init__(self, n_metatype=13, n_embout=256, n_neighbor=100, n_stepk=10):
        '''
        :param n_metatype: 基元类别总数
        :param n_embout: 参数化模板特征提取主干输出特征长度
        :param n_neighbor: 每个点的邻域内点数，包含该点本身
        :param n_stepk: surface_knn 找点时每步的点数
        '''
        super().__init__()

        self.n_neighbor = n_neighbor
        self.n_stepk = n_stepk

        # pointnet++ encoder
        self.backbone_trifea = pointnet2_partseg_encoder()

        # 逐行回归欧拉角的MLP，属于回归
        self.eula_angle = full_connected_conv1d([n_embout, 256, 128, 32, 3])

        # 逐行回归是否属于边界点的MLP，属于分类
        self.edge_nearby = full_connected_conv1d([n_embout, 256, 128, 32, 2])

        # 逐行回归属于何种基元类别的MLP，属于分类
        self.meta_type = full_connected_conv1d([n_embout, 256, 128, 64, n_metatype])

    def forward(self, points_all):
        # points_all: torch.size([bs, n_points_all, 3(6)])
        bs, n_points_all, n_channel = points_all.size()
        device = points_all.device

        ex_features = self.backbone_trifea(points_all)
        # ex_features = self.backbone_trifea(all_assigned_pnts)
        # [bs, n_points_all, self.n_embout]

        ex_features = ex_features[0].transpose(-1, -2)
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


class paramnet_backup1(nn.Module):
    def __init__(self, n_metatype=13, n_classes=2, n_embout=256, n_neighbor=100, n_stepk=10):
        '''
        :param n_metatype: 基元类别总数
        :param n_classes: 总类别数
        :param n_embout: 参数化模板特征提取主干输出特征长度
        :param n_neighbor: 每个点的邻域内点数，包含该点本身
        :param n_stepk: surface_knn 找点时每步的点数
        '''
        super().__init__()

        self.n_neighbor = n_neighbor
        self.n_stepk = n_stepk

        # pointnet++ encoder
        param_l0 = PN2CSEncoderParam()
        param_l0.append(npoint=int(self.n_neighbor/2), radius=0.2, nsample=9, in_channel=3, mlp=[16, 16, 32], group_all=False)
        param_l0.append(npoint=int(self.n_neighbor/4), radius=0.4, nsample=12, in_channel=32 + 3, mlp=[32, 32, 64], group_all=False)
        param_l0.append(npoint=None, radius=None, nsample=None, in_channel=64+3, mlp=[64, 128, n_embout], group_all=True)

        # self.backbone_trifea = mini_encoder(param_l0)
        # self.backbone_trifea = pointnet2_cls_ssg_encoder(param_l0)
        self.backbone_trifea = pointnet2_partseg_encoder()

        # 逐行回归欧拉角的MLP，属于回归
        self.eula_angle = full_connected_conv1d([n_embout, 256, 128, 32, 3])

        # 逐行回归是否属于边界点的MLP，属于分类
        self.edge_nearby = full_connected_conv1d([n_embout, 256, 128, 32, 2])

        # 逐行回归属于何种基元类别的MLP，属于分类
        self.meta_type = full_connected_conv1d([n_embout, 256, 128, 64, n_metatype])

        # 拼接特征后进行分割的网络
        param_l1 = PN2CSEncoderParam()
        param_l1.append(npoint=512, radius=0.2, nsample=32, in_channel=3+3+2+n_metatype, mlp=[64, 64, 128], group_all=False)
        param_l1.append(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        param_l1.append(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)

        self.multiattr_classifier = pointnet2_cls_ssg_encoder(param_l1)
        self.final_type = full_connected([1024, 256, 64, n_classes])

    def forward(self, points_all):
        # points_all: torch.size([bs, n_points_all, 3(6)])
        bs, n_points_all, n_channel = points_all.size()
        device = points_all.device

        # neighbor_surfind_all = surface_knn_all(points_all, self.n_neighbor, self.n_stepk).to(device)
        # # torch.size(bs, n_points_all, self.n_neighbor)
        #
        # # 构造 torch.Size([batch_size, n_points_all, n_points_neighbor, 3]) 矩阵
        # # 从 points_all: torch.size([bs, n_points_all, 3(6)]) 中选 torch.size(bs, n_points_all, self.n_neighbor) 个点
        # all_assigned_pnts = []
        # for i in range(bs):
        #     tmp_all = points_all[i, :, :]
        #     tmp_all = torch.tile(tmp_all, (n_points_all, 1, 1))
        #
        #     tmp_selected = index_points(tmp_all, neighbor_surfind_all[i, :, :])
        #     all_assigned_pnts.append(tmp_selected.unsqueeze(dim=0))
        #
        # all_assigned_pnts = torch.cat(all_assigned_pnts, dim=0)

        ex_features = self.backbone_trifea(points_all)
        # ex_features = self.backbone_trifea(all_assigned_pnts)
        # [bs, n_points_all, self.n_embout]

        ex_features = ex_features[0].transpose(-1, -2)
        # [bs, self.n_embout, n_points_all]

        eula_angle = self.eula_angle(ex_features).transpose(-1, -2)
        # [bs, n_points_all, 3]

        edge_nearby = self.edge_nearby(ex_features).transpose(-1, -2)
        # [bs, n_points_all, 2]

        meta_type = self.meta_type(ex_features).transpose(-1, -2)
        # [bs, n_points_all, 7]

        attr_all = torch.cat(
            (
                points_all,
                eula_angle,
                F.softmax(edge_nearby, dim=-1),
                F.softmax(meta_type, dim=-1)
            ), dim=-1)
        # [bs, n_points_all, 3 + 3 + 2 + 7]

        attr_all = self.multiattr_classifier(attr_all)

        attr_all = self.final_type(attr_all)
        final_type_log = F.log_softmax(attr_all, dim=-1)

        edge_nearby_log = F.log_softmax(edge_nearby, dim=-1)
        meta_type_log = F.log_softmax(meta_type, dim=-1)

        return final_type_log, eula_angle, edge_nearby_log, meta_type_log


class paramnet_test1(nn.Module):
    def __init__(self, n_metatype=13, n_classes=2, n_embout=256, n_neighbor=100, n_stepk=10):
        '''
        :param n_metatype: 基元类别总数
        :param n_classes: 总类别数
        :param n_embout: 参数化模板特征提取主干输出特征长度
        :param n_neighbor: 每个点的邻域内点数，包含该点本身
        :param n_stepk: surface_knn 找点时每步的点数
        '''
        super().__init__()

        # 拼接特征后进行分割的网络
        param_l1 = PN2CSEncoderParam()
        param_l1.append(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        param_l1.append(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        param_l1.append(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)

        self.multiattr_classifier = pointnet2_cls_ssg_encoder(param_l1)
        self.final_type = full_connected([1024, 256, 64, n_classes])

    def forward(self, points_all):

        attr_all = self.multiattr_classifier(points_all)

        attr_all = self.final_type(attr_all)
        final_type_log = F.log_softmax(attr_all, dim=-1)

        return final_type_log


class paramnet_v2(nn.Module):
    '''
    预测额外属性时，输入逐点的邻域
    '''
    def __init__(self, n_metatype: int = 4, n_embout: int = 256, n_neighbor: int = 100, n_stepk: int = 10):
        '''
        :param n_metatype: 基元类别总数
        :param n_embout: 参数化模板特征提取主干输出特征长度
        :param n_neighbor: 每个点的邻域内点数，包含该点本身
        :param n_stepk: surface_knn 找点时每步的点数
        '''
        super().__init__()

        self.n_neighbor = n_neighbor
        self.n_stepk = n_stepk

        # pointnet++ encoder
        param_l0 = PN2CSEncoderParam()
        param_l0.append(npoint=int(self.n_neighbor/2), radius=0.2, nsample=9, in_channel=3, mlp=[16, 16, 32], group_all=False)
        param_l0.append(npoint=int(self.n_neighbor/4), radius=0.4, nsample=12, in_channel=32 + 3, mlp=[32, 32, 64], group_all=False)
        param_l0.append(npoint=None, radius=None, nsample=None, in_channel=64+3, mlp=[64, 128, n_embout], group_all=True)

        self.backbone_trifea = mini_encoder2(param_l0)

        # 逐行回归欧拉角的MLP，属于回归
        self.eula_angle = full_connected([n_embout, 256, 128, 32, 3])

        # 逐行回归是否属于边界点的MLP，属于分类
        self.edge_nearby = full_connected([n_embout, 256, 128, 32, 2])

        # 逐行回归属于何种基元类别的MLP，属于分类
        self.meta_type = full_connected([n_embout, 256, 128, 64, n_metatype])

    def forward(self, points_all):
        # points_all: [bs, n_neighbor, 3]

        ex_features = self.backbone_trifea(points_all)
        # [bs, self.n_embout]

        eula_angle = self.eula_angle(ex_features)
        # [bs, 3]

        edge_nearby = self.edge_nearby(ex_features)
        # [bs, 2]

        meta_type = self.meta_type(ex_features)
        # [bs, n_metatype]

        edge_nearby_log = F.log_softmax(edge_nearby, dim=-1)
        meta_type_log = F.log_softmax(meta_type, dim=-1)

        return eula_angle, edge_nearby_log, meta_type_log


class paramnet_v1(nn.Module):
    def __init__(self, n_metatype=13, n_classes=2, n_embout=256, n_neighbor=100, n_stepk=10):
        '''
        :param n_metatype: 基元类别总数
        :param n_classes: 总类别数
        :param n_embout: 参数化模板特征提取主干输出特征长度
        :param n_neighbor: 每个点的邻域内点数，包含该点本身
        :param n_stepk: surface_knn 找点时每步的点数
        '''
        super().__init__()

        self.n_neighbor = n_neighbor
        self.n_stepk = n_stepk

        # pointnet++ encoder
        param_l0 = PN2CSEncoderParam()
        param_l0.append(npoint=int(self.n_neighbor/2), radius=0.2, nsample=9, in_channel=3, mlp=[16, 16, 32], group_all=False)
        param_l0.append(npoint=int(self.n_neighbor/4), radius=0.4, nsample=12, in_channel=32 + 3, mlp=[32, 32, 64], group_all=False)
        param_l0.append(npoint=None, radius=None, nsample=None, in_channel=64+3, mlp=[64, 128, n_embout], group_all=True)

        self.backbone_trifea = mini_encoder(param_l0)

        # 逐行回归欧拉角的MLP，属于回归
        self.eula_angle = full_connected_conv1d([n_embout, 256, 128, 32, 3])

        # 逐行回归是否属于边界点的MLP，属于分类
        self.edge_nearby = full_connected_conv1d([n_embout, 256, 128, 32, 2])

        # 逐行回归属于何种基元类别的MLP，属于分类
        self.meta_type = full_connected_conv1d([n_embout, 256, 128, 64, n_metatype])

        # 拼接特征后进行分割的网络
        param_l1 = PN2CSEncoderParam()
        param_l1.append(npoint=512, radius=0.2, nsample=32, in_channel=3+3+2+n_metatype, mlp=[64, 64, 128], group_all=False)
        param_l1.append(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        param_l1.append(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)

        self.multiattr_classifier = pointnet2_cls_ssg_encoder(param_l1)
        self.final_type = full_connected([1024, 512, 256, 128, 64, n_classes])

    def forward(self, points_all):
        # points_all: torch.size([bs, n_points_all, 3(6)])
        bs, n_points_all, n_channel = points_all.size()
        device = points_all.device

        neighbor_surfind_all = surface_knn_all(points_all, self.n_neighbor, self.n_stepk).to(device)
        # torch.size(bs, n_points_all, self.n_neighbor)

        # 构造 torch.Size([batch_size, n_points_all, n_points_neighbor, 3]) 矩阵
        # 从 points_all: torch.size([bs, n_points_all, 3(6)]) 中选 torch.size(bs, n_points_all, self.n_neighbor) 个点
        all_assigned_pnts = []
        for i in range(bs):
            tmp_all = points_all[i, :, :]
            tmp_all = torch.tile(tmp_all, (n_points_all, 1, 1))

            tmp_selected = index_points(tmp_all, neighbor_surfind_all[i, :, :])
            all_assigned_pnts.append(tmp_selected.unsqueeze(dim=0))

        all_assigned_pnts = torch.cat(all_assigned_pnts, dim=0)

        ex_features = self.backbone_trifea(all_assigned_pnts)
        # [bs, n_points_all, self.n_embout]

        ex_features = ex_features.transpose(-1, -2)
        # [bs, self.n_embout, n_points_all]

        eula_angle = self.eula_angle(ex_features).transpose(-1, -2)
        # [bs, n_points_all, 3]

        edge_nearby = self.edge_nearby(ex_features).transpose(-1, -2)
        # [bs, n_points_all, 2]

        meta_type = self.meta_type(ex_features).transpose(-1, -2)
        # [bs, n_points_all, 7]

        attr_all = torch.cat(
            (
                points_all,
                eula_angle,
                F.softmax(edge_nearby, dim=-1),
                F.softmax(meta_type, dim=-1)
            ), dim=-1)
        # [bs, n_points_all, 3 + 3 + 2 + 7]

        attr_all = self.multiattr_classifier(attr_all)

        attr_all = self.final_type(attr_all)
        final_type_log = F.log_softmax(attr_all, dim=-1)

        edge_nearby_log = F.log_softmax(edge_nearby, dim=-1)
        meta_type_log = F.log_softmax(meta_type, dim=-1)

        return final_type_log, eula_angle, edge_nearby_log, meta_type_log


def test():
    test_tensor = torch.rand((2, 1000, 3)).cuda()
    classifier = paramnet_v2().cuda()
    _, _, _ = classifier(test_tensor)
    pass


if __name__ == '__main__':
    test()

