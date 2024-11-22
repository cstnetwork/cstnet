import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
import torch


class get_model(nn.Module):
    # num_class = 40，normal_channel = false
    def __init__(self, num_class: int, normal_channel: bool=False, addattr_channel: int=0, is_useeula: bool=False, is_usenearby: bool=False, is_usemetatype: bool=False):
        super(get_model, self).__init__()
        assert addattr_channel >= 0
        if addattr_channel is not 0:
            self.is_addattr = True
        else:
            self.is_addattr = False

        self.is_useeula = is_useeula
        self.is_usenearby = is_usenearby
        self.is_usemetatype = is_usemetatype

        # in_channel = 3，输入的通道数，通道为6表示使用法向量
        in_channel = 6 if normal_channel else 3

        if self.is_useeula:
            in_channel = in_channel + 3
        elif self.is_usenearby:
            in_channel = in_channel + 2
        elif self.is_usemetatype:
            in_channel = in_channel + addattr_channel - 3 - 2
        else:
            in_channel = in_channel + addattr_channel

        # false
        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz, add_attr=None):
        B, _, _ = xyz.shape

        if self.is_addattr:
            if self.normal_channel:
                norm = xyz[:, 3:, :]
                xyz = xyz[:, :3, :]
                if add_attr is not None:
                    norm = torch.cat([norm, add_attr], dim=-2)
            else:
                norm = None
                if add_attr is not None:
                    if self.is_useeula:
                        norm = add_attr[:, 0: 3, :]
                    elif self.is_usenearby:
                        norm = add_attr[:, 3: 5, :]
                    elif self.is_usemetatype:
                        norm = add_attr[:, 5:, :]
                    else:
                        norm = add_attr

        else:
            if self.normal_channel:
                norm = xyz[:, 3:, :]
                xyz = xyz[:, :3, :]
            else:
                norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
