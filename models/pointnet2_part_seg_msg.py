import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel: # normal_channel = False
            additional_channel = 3
        else:
            additional_channel = 0
        # additional_channel = 0

        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        '''
            xyz: torch.Size([batch_size, 3, n_points]) torch.Size([16, 3, 2048])
            cls_label: torch.Size([batch_size, 1, n_object_class])
            cls_label 的每个批量为 1 行 n_object_class 列的向量，n_object_class为总的分类类别数
                    相当于使用 one-hot vector 代表类别
        '''

        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel: # normal_channel = False
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        # l1_xyz: 采样后获得的点 torch.Size([16, 3, 512])
        # l1_points 多尺度特征提取后获得的特征向量 torch.Size([16, 320, 512])
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # l1_points: torch.Size([16, 128, 512])

        # cls_label: torch.Size([batch_size, 1, n_object_class])
        #       |0, 0, 1, 0, 0, 0, 0|
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        # cls_label_one_hot: torch.Size([batch_size, n_object_class, n_points])
        #       —— —— —— ——
        #       0  0  0  0
        #       0  0  0  0
        #       1  1  1  1
        #       0  0  0  0
        #       0  0  0  0
        #       0  0  0  0
        #       0  0  0  0
        #       —— —— —— ——
        #       以上例子假设点云中所有点总数为 4, 物体类别总数为 7

        # torch.cat([cls_label_one_hot,l0_xyz,l0_points],1): torch.Size([16, 22, 2048])
        #       cls_label_one_hot: torch.Size([16, 16, 2048])
        #       l0_xyz: torch.Size([16, 3, 2048]) 输入的点云坐标数据 = xyz
        #       l0_points: torch.Size([16, 3, 2048]) 输入的点云坐标数据 = xyz, 因此 l0_xyz 和 l0_points 是一样的
        #       —— —— —— ——
        #       0  0  0  0       cls_label_one_hot
        #       0  0  0  0
        #       1  1  1  1
        #       0  0  0  0
        #       0  0  0  0
        #       0  0  0  0
        #       0  0  0  0
        #
        #       x1 x2 x3 x4       l0_xyz
        #       y1 y2 y3 y4
        #       z1 z2 z3 z4
        #
        #       x1 x2 x3 x4       l0_points
        #       y1 y2 y3 y4
        #       z1 z2 z3 z4
        #       —— —— —— ——

        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss