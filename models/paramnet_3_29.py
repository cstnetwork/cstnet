# 工具包
import torch.nn as nn
import torch
import torch.nn.functional as F

# 自定义模块
from utils import surface_knn_all
from utils import full_connected_conv1d, full_connected
from encoders import PN2CSEncoderParam
from encoders import pointnet2_cls_ssg_encoder, mini_encoder
from pointnet2_utils import index_points

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

        return final_type_log, eula_angle, edge_nearby, meta_type


def test():
    test_tensor = torch.rand((2, 1000, 3)).cuda()
    classifier = paramnet().cuda()
    cls, _, _, _ = classifier(test_tensor)
    pass


if __name__ == '__main__':
    test()

