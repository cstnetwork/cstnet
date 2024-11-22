import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from pointnet2_utils import PointNetSetAbstraction, PointNetSetAbstractionAllVert
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation
import utils


class PN2CSEncoderParam(object):
    def __init__(self):
        self.n_groups = []
        self.ball_radius = []
        self.n_ballsamples = []
        self.n_channelin = []
        self.mlps = []
        self.group_all = []

    def append(self, npoint, radius, nsample, in_channel, mlp, group_all):
        self.n_groups.append(npoint)
        self.ball_radius.append(radius)
        self.n_ballsamples.append(nsample)
        self.n_channelin.append(in_channel)
        self.mlps.append(mlp)
        self.group_all.append(group_all)

    def __len__(self):
        return len(self.n_groups)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


class mini_encoder(nn.Module):
    def __init__(self, params: PN2CSEncoderParam):
        super().__init__()

        self.set_abstractions = nn.ModuleList()
        self.n_layers = len(params)
        self.n_channelout = params.mlps[-1][-1]

        self.normal_channel = False

        for i in range(self.n_layers):
            self.set_abstractions.append(PointNetSetAbstractionAllVert(
                n_center_fps=params.n_groups[i],
                radius=params.ball_radius[i],
                n_sample_ball=params.n_ballsamples[i],
                n_channel_in=params.n_channelin[i],
                mlp=params.mlps[i],
                is_group_all=params.group_all[i])
            )

        self.apply(inplace_relu)

    def forward(self, xyz):
        # xyz: torch.Size(batch_size, n_points_all, n_near, 3)
        bs, n_points_all, _, _ = xyz.shape

        if self.normal_channel:
            norm = xyz[:, :, :, 3:]
            xyz = xyz[:, :, :, :3]
        else:
            norm = None

        l_xyz, l_points = xyz, norm
        for i in range(self.n_layers):
            l_xyz, l_points = self.set_abstractions[i](l_xyz, l_points)

        x = l_points.view(bs, n_points_all, self.n_channelout)

        return x


class mini_encoder2(nn.Module):
    def __init__(self, params: PN2CSEncoderParam):
        super().__init__()

        self.set_abstractions = nn.ModuleList()
        self.n_layers = len(params)
        self.n_channelout = params.mlps[-1][-1]

        self.normal_channel = False

        for i in range(self.n_layers):
            self.set_abstractions.append(PointNetSetAbstraction(npoint=params.n_groups[i],
                                                                radius=params.ball_radius[i],
                                                                nsample=params.n_ballsamples[i],
                                                                in_channel=params.n_channelin[i],
                                                                mlp=params.mlps[i],
                                                                group_all=params.group_all[i])
                                         )

        self.apply(inplace_relu)

    def forward(self, xyz):
        bs, n_points, n_channels = xyz.shape

        if n_channels > 3:
            features = xyz[:, :, 3:]
            features = features.transpose(-2, -1)
            xyz = xyz[:, :, :3]

        else:
            features = None

        xyz = xyz.transpose(-2, -1)

        l_xyz, l_points = xyz, features
        # l_xyz: [bs, 3, n_pnts], l_points: [bs, channel, n_pnts]

        for i in range(self.n_layers):
            l_xyz, l_points = self.set_abstractions[i](l_xyz, l_points)

        x = l_points.view(bs, self.n_channelout)

        return x


class MiniEncoder_PN2Seg(nn.Module):
    def __init__(self, n_outchanell):
        super().__init__()

        # additional_channel = 0
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, n_outchanell, 1)

    def forward(self, xyz):
        # Set Abstraction layers
        xyz = xyz.transpose(2, 1)
        B,C,N = xyz.shape

        l0_points = xyz
        l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # l1_points: torch.Size([16, 128, 512])

        # cls_label: torch.Size([batch_size, 1, n_object_class])
        #       |0, 0, 1, 0, 0, 0, 0|
        cls_label = torch.zeros(B,16).to(xyz.device)  # ---------------------------------------------------------
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)


        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers

        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        return x


class AttentionTriPN2(nn.Module):
    def __init__(self, normal_channel=False):
        super().__init__()
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
        self.conv1 = nn.Conv1d(128, 256, 1)

    def forward(self, xyz):
        xyz = xyz.transpose(2, 1)
        B,C,N = xyz.shape
        if self.normal_channel: # normal_channel = False
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # l1_points: torch.Size([16, 128, 512])

        # cls_label: torch.Size([batch_size, 1, n_object_class])
        #       |0, 0, 1, 0, 0, 0, 0|
        cls_label = torch.zeros(B,16).to(xyz.device)  # ---------------------------------------------------------
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)

        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        x = self.conv1(l0_points)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class PN2PartSegSsgEncoder(nn.Module):
    '''
    pointnet++ seg ssg Encoder
    '''
    def __init__(self, channel_out=256):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3+3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])  # in_chanell = points2_chanell + points1_channel
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+6, mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, channel_out, 1)

    def forward(self, xyz):
        # -> xyz: [bs, n_points, 3]
        xyz = xyz.transpose(1, -1)

        B,C,N = xyz.shape

        # Set Abstraction layers
        l0_points = xyz
        l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)

        # FC layers
        feat = self.conv1(l0_points)
        feat = feat.permute(0, 2, 1)

        return feat


class PN2PartSegMsgEncoder(nn.Module):
    '''
    pointnet++ seg msg Encoder
    '''
    def __init__(self, channel_out=256):
        super().__init__()

        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, channel_out, 1)

    def forward(self, xyz):
        xyz = xyz.transpose(1, -1)
        # <- xyz: [bs, n_points, 3], -> [bs, 3, n_points]

        l0_points = xyz
        l0_xyz = xyz

        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)

        # FC layers
        feat = self.conv1(l0_points)
        feat = feat.permute(0, 2, 1)

        return feat


class PN2PartSegSsgEncoder_AttentionV5(nn.Module):
    def __init__(self, n_points_all, rate_downsample=0.9, channel_out=256):
        super().__init__()

        self.sa1 = utils.SA_Attention_test3(n_center=int(n_points_all * rate_downsample), n_near=32, dim_in=3, dim_qkv=32, mlp=[32, 64, 128])
        self.sa2 = utils.SA_Attention_test3(n_center=int(n_points_all * rate_downsample ** 2), n_near=64, dim_in=128 + 3, dim_qkv=128, mlp=[128, 128, 256])

        self.fp2 = utils.FeaPropagate(in_channel=(256+128+3)+(128+3), mlp=[256, 128])  # in_chanell = points2_chanell + points1_channel
        self.fp1 = utils.FeaPropagate(in_channel=128+(3+3), mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, channel_out, 1)

    def forward(self, xyz):
        # -> xyz: [bs, n_points, 3]
        # Set Abstraction layers
        l0_points = xyz
        l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        l0_points = l0_points.permute(0, 2, 1)
        l0_xyz = l0_xyz.permute(0, 2, 1)
        l1_xyz = l1_xyz.permute(0, 2, 1)
        l1_points = l1_points.permute(0, 2, 1)
        l2_xyz = l2_xyz.permute(0, 2, 1)
        l2_points = l2_points.permute(0, 2, 1)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)

        # FC layers
        feat = self.conv1(l0_points)
        feat = feat.permute(0, 2, 1)

        return feat


class PN2PartSegSsgEncoder_AttentionV4(nn.Module):
    def __init__(self, n_points_all, rate_downsample=0.9, channel_out=256):
        super().__init__()

        self.sa1 = utils.SA_Attention_test2(n_center=int(n_points_all * rate_downsample), n_near=32, dim_in=3, dim_qk=16, dim_v=32, mlp=[32, 64, 128])
        self.sa2 = utils.SA_Attention_test2(n_center=int(n_points_all * rate_downsample ** 2), n_near=64, dim_in=128 + 3, dim_qk=128, dim_v=128, mlp=[128, 128, 256])

        self.fp2 = utils.FeaPropagate(in_channel=(256+128+3)+(128+3), mlp=[256, 128])  # in_chanell = points2_chanell + points1_channel
        self.fp1 = utils.FeaPropagate(in_channel=128+(3+3), mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, channel_out, 1)

    def forward(self, xyz):
        # -> xyz: [bs, n_points, 3]

        # Set Abstraction layers
        l0_points = xyz
        l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        # Feature Propagation layers
        # fp 层输入都是 [bs, channel, n_points]
        l0_points = l0_points.permute(0, 2, 1)
        l0_xyz = l0_xyz.permute(0, 2, 1)
        l1_xyz = l1_xyz.permute(0, 2, 1)
        l1_points = l1_points.permute(0, 2, 1)
        l2_xyz = l2_xyz.permute(0, 2, 1)
        l2_points = l2_points.permute(0, 2, 1)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)

        # FC layers
        feat = self.conv1(l0_points)
        feat = feat.permute(0, 2, 1)

        return feat


class PN2PartSegSsgEncoder_AttentionV3(nn.Module):
    def __init__(self, n_points_all, rate_downsample=0.9, channel_out=256):
        super().__init__()

        self.sa1 = utils.SA_Attention_test1(n_center=int(n_points_all * rate_downsample), n_near=32, dim_in=3, dim_qkv=32, mlp=[32, 64, 128])
        self.sa2 = utils.SA_Attention_test1(n_center=int(n_points_all * rate_downsample ** 2), n_near=64, dim_in=128 + 3, dim_qkv=128, mlp=[128, 128, 256])

        self.fp2 = utils.FeaPropagate(in_channel=(256+128+3)+(128+3), mlp=[256, 128])  # in_chanell = points2_chanell + points1_channel
        self.fp1 = utils.FeaPropagate(in_channel=128+(3+3), mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, channel_out, 1)

    def forward(self, xyz):
        # -> xyz: [bs, n_points, 3]

        # Set Abstraction layers
        l0_points = xyz
        l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        l0_points = l0_points.permute(0, 2, 1)
        l0_xyz = l0_xyz.permute(0, 2, 1)
        l1_xyz = l1_xyz.permute(0, 2, 1)
        l1_points = l1_points.permute(0, 2, 1)
        l2_xyz = l2_xyz.permute(0, 2, 1)
        l2_points = l2_points.permute(0, 2, 1)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)

        # FC layers
        feat = self.conv1(l0_points)
        feat = feat.permute(0, 2, 1)

        return feat


class PN2PartSegSsgEncoder_AttentionV2(nn.Module):
    def __init__(self, n_points_all, rate_downsample=0.9, channel_out=256):
        super().__init__()

        # self.sa1 = utils.SA_Scratch(n_center=None, n_near=128, in_channel=3+3, mlp=[64, 64, 128])
        # self.sa2 = utils.SA_Scratch(n_center=int(n_points_all * rate_downsample), n_near=64, in_channel=128+3, mlp=[128, 128, 256])
        self.sa1 = utils.SA_Attention(n_center=int(n_points_all * rate_downsample), n_near=50, in_channel=3+3, mlp=[64, 64, 128])
        self.sa2 = utils.SA_Attention(n_center=int(n_points_all * rate_downsample ** 2), n_near=100, in_channel=128+3, mlp=[128, 128, 256])

        self.fp2 = utils.FeaPropagate(in_channel=256+128, mlp=[256, 256, 128])  # in_chanell = points2_chanell + points1_channel
        self.fp1 = utils.FeaPropagate(in_channel=128+6, mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, channel_out, 1)

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
        feat = feat.permute(0, 2, 1)

        return feat


class PN2PartSegSsgEncoder_AttentionV1(nn.Module):
    def __init__(self, n_points_all, rate_downsample=0.9, channel_out=256):
        super().__init__()

        self.sa1 = utils.SA_Scratch(n_center=int(n_points_all * rate_downsample), n_near=32, in_channel=3+3, mlp=[64, 64, 128])
        self.sa2 = utils.SA_Scratch(n_center=int(n_points_all * rate_downsample ** 2), n_near=64, in_channel=128+3, mlp=[128, 128, 256])

        self.fp2 = utils.FeaPropagate(in_channel=384, mlp=[256, 128])  # in_chanell = points2_chanell + points1_channel
        self.fp1 = utils.FeaPropagate(in_channel=128+6, mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, channel_out, 1)

    def forward(self, xyz):
        # -> xyz: [bs, n_points, 3]
        xyz = xyz.transpose(1, -1)

        B,C,N = xyz.shape

        # Set Abstraction layers
        l0_points = xyz
        l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        # Feature Propagation layers
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)

        # FC layers
        feat = self.conv1(l0_points)
        feat = feat.permute(0, 2, 1)

        return feat


class pointnet2_partseg_encoder(nn.Module):
    '''
    part seg msg
    '''
    def __init__(self, normal_channel=False):
        super().__init__()
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
        self.conv1 = nn.Conv1d(128, 256, 1)

    def forward(self, xyz):
        xyz = xyz.transpose(2, 1)
        B,C,N = xyz.shape
        if self.normal_channel: # normal_channel = False
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # l1_points: torch.Size([16, 128, 512])

        # cls_label: torch.Size([batch_size, 1, n_object_class])
        #       |0, 0, 1, 0, 0, 0, 0|
        cls_label = torch.zeros(B,16).to(xyz.device)  # ---------------------------------------------------------
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)

        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        x = self.conv1(l0_points)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class pointnet2_cls_ssg_encoder(nn.Module):
    def __init__(self, params: PN2CSEncoderParam):
        super().__init__()

        self.set_abstractions = nn.ModuleList()
        self.n_layers = len(params)
        self.n_channelout = params.mlps[-1][-1]

        self.normal_channel = False

        for i in range(self.n_layers):
            self.set_abstractions.append(PointNetSetAbstraction(npoint=params.n_groups[i],
                                                                radius=params.ball_radius[i],
                                                                nsample=params.n_ballsamples[i],
                                                                in_channel=params.n_channelin[i],
                                                                mlp=params.mlps[i],
                                                                group_all=params.group_all[i])
                                         )

        self.apply(inplace_relu)

    def forward(self, xyz):
        bs, n_points, n_channels = xyz.shape

        if n_channels > 3:
            features = xyz[:, :, 3:]
            features = features.transpose(-2, -1)
            xyz = xyz[:, :, :3]

        else:
            features = None

        xyz = xyz.transpose(-2, -1)

        l_xyz, l_points = xyz, features
        for i in range(self.n_layers):
            l_xyz, l_points = self.set_abstractions[i](l_xyz, l_points)

        x = l_points.view(bs, self.n_channelout)

        return x


class PointAttentionEncoder1D(nn.Module):
    def __init__(self, process_channels: list = [3, 16, 32]):
        super().__init__()

        out_channel = process_channels[-1]

        self.center_q = utils.full_connected(process_channels)
        self.center_k = utils.full_connected(process_channels)
        self.center_v = utils.full_connected(process_channels)

        self.other_k = utils.full_connected_conv1d(process_channels)
        self.other_v = utils.full_connected_conv1d(process_channels)

        self.pos = utils.full_connected_conv1d(process_channels)
        self.weight_process = utils.full_connected_conv1d([out_channel, 2 * out_channel, out_channel])

    def forward(self, center, other):
        fea_centerq = self.center_q(center)  # ->[bs, out_channel]
        fea_centerk = self.center_k(center)  # ->[bs, out_channel]
        fea_centerv = self.center_v(center)  # ->[bs, out_channel]

        fea_otherk = self.other_k(other)  # ->[bs, n_nearby, out_channel]
        fea_otherv = self.other_v(other)  # ->[bs, n_nearby, out_channel]
        fea_otherk = torch.cat((fea_centerk, fea_otherk), dim=-2)  # ->[bs, n_nearby + 1, out_channel]
        fea_otherv = torch.cat((fea_centerv, fea_otherv), dim=-2)  # ->[bs, n_nearby + 1, out_channel]

        fea_minus = center - torch.cat((center, other), dim=-2)  # ->[bs, n_nearby + 1, out_channel]
        fea_pos = self.pos(fea_minus)  # ->[bs, n_nearby + 1, out_channel]

        # q - k
        q_minus_k = fea_centerq - fea_otherk  # ->[bs, n_nearby + 1, out_channel]

        # mlp(q - k + delta)
        fea_weight = self.weight_process(q_minus_k + fea_pos)  # ->[bs, n_nearby + 1, out_channel]

        # softmax(mlp(q - k + pos)) * (v + pos)
        weight_attention = F.softmax(fea_weight, dim=-1) * (fea_otherv + fea_pos)  # ->[bs, n_nearby + 1, out_channel]

        # sum(softmax(mlp(q - k + pos)) * (v + pos))
        fea_out = torch.sum(weight_attention, dim=-2)  # ->[bs, out_channel]

        return fea_out


class PointAttentionEncoder2D(nn.Module):
    def __init__(self, process_channels: list = [3, 16, 32]):
        super().__init__()

        out_channel = process_channels[-1]

        self.center_q = utils.full_connected_conv1d(process_channels)
        self.center_k = utils.full_connected_conv1d(process_channels)
        self.center_v = utils.full_connected_conv1d(process_channels)

        self.other_k = utils.full_connected_conv2d(process_channels)
        self.other_v = utils.full_connected_conv2d(process_channels)

        self.pos = utils.full_connected_conv2d(process_channels)
        self.weight_process = utils.full_connected_conv2d([out_channel, 2 * out_channel, out_channel])

    def forward(self, center, other):
        center = center.transpose(-1, 1)  # ->[bs, channels_in, n_center]
        fea_centerq = self.center_q(center)  # ->[bs, channels_out, n_center]
        fea_centerk = self.center_k(center)  # ->[bs, channels_out, n_center]
        fea_centerv = self.center_v(center)  # ->[bs, channels_out, n_center]

        other = other.transpose(-1, 1)  # ->[bs, channels_in, n_nearby, n_center]
        fea_otherk = self.other_k(other)  # ->[bs, channels_out, n_nearby, n_center]
        fea_otherv = self.other_v(other)  # ->[bs, channels_out, n_nearby, n_center]
        fea_otherk = torch.cat((fea_centerk.unsqueeze(2), fea_otherk), dim=-2)  # ->[bs, channels_out, n_nearby + 1, n_center]
        fea_otherv = torch.cat((fea_centerv.unsqueeze(2), fea_otherv), dim=-2)  # ->[bs, channels_out, n_nearby + 1, n_center]
        fea_otherv = fea_otherv.transpose(-1, 1)  # ->[bs, n_center, n_nearby + 1, channels_out]

        fea_minus = center.unsqueeze(2) - torch.cat((center.unsqueeze(2), other), dim=-2)  # ->[bs, channels_in, n_nearby + 1, n_center]
        fea_pos = self.pos(fea_minus)  # ->[bs, channels_out, n_nearby + 1, n_center]

        # q - k
        q_minus_k = fea_centerq.unsqueeze(2) - fea_otherk  # ->[bs, channels_out, n_nearby + 1, n_center]

        # mlp(q - k + delta)
        fea_weight = self.weight_process(q_minus_k + fea_pos)  # ->[bs, channels_out, n_nearby + 1, n_center]
        fea_weight = fea_weight.transpose(-1, 1)  # ->[bs, n_center, n_nearby + 1, channels_out]

        # softmax(mlp(q - k + pos)) * (v + pos)
        fea_pos = fea_pos.transpose(-1, 1)  # ->[bs, n_center, n_nearby + 1, channels_out]
        weight_attention = F.softmax(fea_weight, dim=-2) * (fea_otherv + fea_pos)  # ->[bs, n_center, n_nearby + 1, channels_out]

        # sum(softmax(mlp(q - k + pos)) * (v + pos))
        fea_out = torch.sum(weight_attention, dim=-2)  # ->[bs, n_center, channels_out]

        return fea_out


class AddAttr_AttentionLayer(nn.Module):
    def __init__(self, process_channels: list = [3, 16, 32], otherother_rate: float = 0.2, downsample_rate: float = 0.8):
        super().__init__()

        self.downsample_rate = downsample_rate
        self.otherother_rate = otherother_rate

        self.attention_layer = PointAttentionEncoder2D(process_channels)

    def forward(self, center, other):
        bs, n_other, _ = other.size()

        center_fit = center.unsqueeze(1)
        other_fit = other.unsqueeze(1)
        fea_center = self.attention_layer(center_fit, other_fit)  # ->[bs, 1, channels_out]

        k_other = math.ceil(n_other * self.otherother_rate)
        if k_other < 3:
            k_other = 3
        indexes_near = utils.get_neighbor_index(other, k_other)
        other_of_other = utils.index_points(other, indexes_near)

        fea_other = self.attention_layer(other, other_of_other)  # ->[bs, n_nearby, channels_out]

        dowmsample_to = math.ceil(self.downsample_rate * n_other)
        dist_ab = fea_center * fea_other  # ->[bs, n_nearby, channels_out]
        dist_a2 = fea_center ** 2  # ->[bs, 1, channels_out]
        dist_b2 = fea_other ** 2  # ->[bs, n_nearby, channels_out]
        dist_all = dist_a2 - 2 * dist_ab + dist_b2  # ->[bs, n_nearby, channels_out]
        dist_all = torch.sum(dist_all, dim=-1)  # ->[bs, n_nearby]
        neighbor_index = torch.topk(dist_all, k=dowmsample_to, dim=-1, largest=False)[1]  # ->[bs, dowmsample_to]

        new_others = utils.index_points(fea_other, neighbor_index)  # ->[bs, dowmsample_to, channels_out]

        return fea_center.squeeze(), new_others


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


def test():

    test_tensor = torch.rand(2, 100, 3).cuda()
    test_btensor = torch.rand(2, 3).cuda()

    testnet = AddAttr_AttentionLayer().cuda()

    forward_res = testnet(test_btensor, test_tensor)

    print(forward_res[0].size(), forward_res[1].size())

def test_ind_pnts():
    test_tensor = torch.rand(2, 10, 3)
    test_ind_tensor = torch.tensor([[[1, 2, 3, 7, 9],
                                    [1, 2, 3, 7, 9],
                                    [1, 2, 3, 7, 9],
                                    [1, 2, 3, 7, 9],
                                    [1, 2, 3, 7, 9],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [2, 2, 2, 2, 2],
                                    [2, 2, 2, 2, 2]],

                                    [[1, 2, 3, 7, 9],
                                    [1, 2, 3, 7, 9],
                                    [1, 2, 3, 7, 9],
                                    [1, 2, 3, 7, 9],
                                    [1, 2, 3, 7, 9],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [2, 2, 2, 2, 2],
                                    [2, 2, 2, 2, 2]]
                                    ])

    index_res = utils.index_points(test_tensor, test_ind_tensor)

    bs = 1
    ind_center = 2

    print(test_tensor[bs, test_ind_tensor[bs, ind_center, :], :])
    print(index_res[bs, ind_center, :, :])


if __name__ == '__main__':
    test()
    # test_ind_pnts()


