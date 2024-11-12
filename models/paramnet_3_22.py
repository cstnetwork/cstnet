# 工具包
import torch.nn as nn
import torch
import torch.nn.functional as F

# 自定义模块
from utils import surface_knn_all
from utils import full_connected
from encoders import PN2CSEncoder
from encoders import pointnet2_cls_ssg_encoder


class paramnet(nn.Module):
    def __init__(self, n_metatype=7, n_classes=2, n_embout=256, n_neighbor=100, n_stepk=10):
        super().__init__()

        self.n_neighbor = n_neighbor
        self.n_stepk = n_stepk
        self.n_embout = n_embout
        self.n_metatype = n_metatype

        # pointnet++ encoder
        param_l0 = PN2CSEncoder()
        param_l0.append(npoint=int(self.n_neighbor/2), radius=0.2, nsample=9, in_channel=3, mlp=[16, 16, 32], group_all=False)
        param_l0.append(npoint=int(self.n_neighbor/4), radius=0.4, nsample=12, in_channel=32 + 3, mlp=[32, 32, 64], group_all=False)
        param_l0.append(npoint=None, radius=None, nsample=None, in_channel=64+3, mlp=[64, 128, self.n_embout], group_all=True)

        self.backbone_trifea = pointnet2_cls_ssg_encoder(param_l0)

        # 逐行回归欧拉角的MLP，属于回归
        self.eula_angle = full_connected([self.n_embout, 256, 128, 32, 3])

        # 逐行回归是否属于边界点的MLP，属于分类
        self.edge_nearby = full_connected([self.n_embout, 256, 128, 32, 2])

        # 逐行回归属于何种基元类别的MLP，属于分类
        self.meta_type = full_connected([self.n_embout, 256, 128, 64, self.n_metatype])

        # 拼接特征后进行分割的网络
        param_l1 = PN2CSEncoder()
        param_l1.append(npoint=512, radius=0.2, nsample=32, in_channel=3+3+2+self.n_metatype, mlp=[64, 64, 128], group_all=False)
        param_l1.append(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        param_l1.append(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)

        self.multiattr_classifier = pointnet2_cls_ssg_encoder(param_l1)
        self.final_type = full_connected([1024, 512, 256, 128, 64, n_classes])


    def forward(self, points_all):
        # points_all: torch.size([bs, n_points, 3(6)])
        bs, n_points_all, n_channel = points_all.size()
        device = points_all.device

        neighbor_surfind_all = surface_knn_all(points_all, self.n_neighbor, self.n_stepk).to(device)
        # torch.size(bs, n_points_all, self.n_neighbor)

        ex_features = torch.zeros(bs, n_points_all, self.n_embout).to(device)

        ula_angle_all = torch.zeros(bs, n_points_all, 3).to(device)
        edge_nearby_all = torch.zeros(bs, n_points_all, 2).to(device)
        meta_type_all = torch.zeros(bs, n_points_all, self.n_metatype).to(device)

        for i in range(n_points_all):
            # 先获取所有batch上第 i 个点对应的所有邻近点坐标 torch.size(bs, self.n_neighbor, 3)
            # current_nears = torch.index_select(points_all, 1, neighbor_surfind_all[:, i, :])
            current_nears = torch.zeros(bs, self.n_neighbor, 3).to(device)
            for j in range(bs):
                current_nears[j, :, :] = torch.index_select(points_all[j, :, :], dim=0, index=neighbor_surfind_all[j, i, :])

            ex_features[:, i, :] = self.backbone_trifea(current_nears)

            eula_angle = self.eula_angle(ex_features[:, i, :])
            edge_nearby = self.edge_nearby(ex_features[:, i, :])
            meta_type = self.meta_type(ex_features[:, i, :])

            ula_angle_all[:, i, :] = eula_angle
            edge_nearby_all[:, i, :] = edge_nearby
            meta_type_all[:, i, :] = meta_type

        attr_all = torch.cat(
            (
                points_all,
                ula_angle_all,
                F.softmax(edge_nearby_all, dim=-1),
                F.softmax(meta_type_all, dim=-1)
            ), dim=-1)

        attr_all = self.multiattr_classifier(attr_all)
        attr_all = self.final_type(attr_all)
        final_type_log = F.log_softmax(attr_all, dim=-1)

        return final_type_log, ula_angle_all, edge_nearby_all, meta_type_all


def test():
    test_tensor = torch.rand((2, 1000, 3)).cuda()
    classifier = paramnet().cuda()
    cls, _, _, _ = classifier(test_tensor)
    pass


if __name__ == '__main__':
    test()

