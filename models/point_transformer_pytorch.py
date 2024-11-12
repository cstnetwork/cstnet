import torch
from torch import nn, einsum
from einops import repeat
import torch.nn.functional as F

import utils

# helpers

def exists(val):
    return val is not None

def max_value(t):
    return torch.finfo(t.dtype).max

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

# classes

class PointTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        dim_in,
        dim_out,
        pos_mlp_hidden_dim=64,
        attn_mlp_hidden_mult=4,
        num_neighbors=None
    ):
        '''
        :param dim_in: 输入特征的维度
        :param dim_out: 输出特征的维度
        :param pos_mlp_hidden_dim: 位置编码线性层中隐含层的维度 3->hidden_dim->dim_out
        :param attn_mlp_hidden_mult: 生成注意力权重的线性层隐含层维度倍数 dim->dim*mult->dim
        :param num_neighbors: 每个点以附近的 num_neighbors 更新权重
        '''
        super().__init__()
        self.num_neighbors = num_neighbors

        self.to_qkv = nn.Linear(dim_in, dim_out * 3, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim_out)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim_out, dim_out * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim_out * attn_mlp_hidden_mult, dim_out),
        )

    def forward(self, x, pos, mask=None):
        '''
        :param x: 输入点的特征 [bs, n_point, dim_in]
        :param pos: 输入点的坐标 [bs, n_point, 3]
        :param mask: 是否忽视某些点 [bs, n_points] bool
        :return: 更新后的点属性 [bs, n_point, dim_out]
        '''
        n, num_neighbors = x.shape[1], self.num_neighbors

        # get queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # -> [bs, n_point, dim_out]

        # calculate relative positional embeddings
        rel_pos_emb = self.pos_mlp(pos)  # ->[bs, n_point, dim_out]

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        # qk_rel = q[:, :, None, :] - k[:, None, :, :]  # 特征两两作差 ->[bs, n_point, n_point, dim_out]

        # prepare mask
        if exists(mask):
            mask = mask[:, :, None] * mask[:, None, :]

        # expand values
        # v = repeat(v, 'b j d -> b i j d', i=n)  # ->[bs, n_point, n_point, dim_out]

        # determine k nearest neighbors for each point, if specified
        if exists(num_neighbors) and num_neighbors < n:
            inner = torch.bmm(pos, pos.transpose(1, 2))  # (bs, v, v)
            quadratic = torch.sum(pos ** 2, dim=2)  # (bs, v)
            rel_dist = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)

            if exists(mask):
                mask_value = max_value(rel_dist)
                rel_dist.masked_fill_(~mask, mask_value)  # ~ 表示取反，True变为False，False变为True

            dist, indices = rel_dist.topk(num_neighbors, largest=False)  # ->[bs, n_point, num_neighbors]

            v = utils.index_points(v, indices)  # ->[bs, n_point, num_neighbors, dim_out]
            q = utils.index_points(q, indices)  # ->[bs, n_point, num_neighbors, dim_out]
            k = utils.index_points(k, indices)

            qk_rel = q - k
            rel_pos_emb = utils.index_points(rel_pos_emb, indices)

            # rel_pos_emb = batched_index_select(rel_pos_emb, indices, dim=2)  # ->[bs, n_point, num_neighbors, dim_out]
            mask = batched_index_select(mask, indices, dim=2) if exists(mask) else None

        # add relative positional embeddings to value
        v = v + rel_pos_emb  # ->[bs, n_point, num_neighbors, dim_out]

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb)  # ->[bs, n_point, num_neighbors, dim_out]

        # masking
        if exists(mask):
            mask_value = -max_value(sim)
            sim.masked_fill_(~mask[..., None], mask_value)

        # attention
        attn = sim.softmax(dim=-2)  # ->[bs, n_point, num_neighbors, dim_out]

        # aggregate
        agg = einsum('b i j d, b i j d -> b i d', attn, v)  # ->[bs, n_point, dim_out]
        return agg


class AttentionCls(nn.Module):
    def __init__(self, n_classes, n_chanellin=3):
        super().__init__()

        self.n_chanellin = n_chanellin

        self.attention1 = PointTransformerLayer(dim_in=n_chanellin,
                                                dim_out=32,
                                                pos_mlp_hidden_dim=64,
                                                attn_mlp_hidden_mult=4,
                                                num_neighbors=32
                                                )

        # 之后FPS取一半的点
        self.attention2 = PointTransformerLayer(dim_in=32,
                                                dim_out=64,
                                                pos_mlp_hidden_dim=64,
                                                attn_mlp_hidden_mult=2,
                                                num_neighbors=32
                                                )

        # 之后FPS取一半的点
        self.attention3 = PointTransformerLayer(dim_in=64,
                                                dim_out=128,
                                                pos_mlp_hidden_dim=64,
                                                attn_mlp_hidden_mult=2,
                                                num_neighbors=64
                                                )

        # 之后FPS取一半的点
        self.attention4 = PointTransformerLayer(dim_in=128,
                                                dim_out=256,
                                                pos_mlp_hidden_dim=64,
                                                attn_mlp_hidden_mult=2,
                                                num_neighbors=64
                                                )

        # 之后 max_pooling
        self.final_pred = utils.full_connected([256, 128, 64, n_classes])

    def forward(self, xyz, fea=None):
        if fea is None:
            assert self.n_chanellin == 3
            fea = xyz

        _, n_pnt, _ = fea.size()

        # 更新特征
        fea = self.attention1(fea, xyz)

        # FPS取1/2的点
        sample_to = int(n_pnt * 0.5)
        idx_fps = utils.farthest_point_sample(xyz, sample_to)
        xyz = utils.index_points(xyz, idx_fps)
        fea = utils.index_points(fea, idx_fps)

        # 更新特征
        fea = self.attention2(fea, xyz)

        # FPS取1/2的点
        sample_to = int(n_pnt * 0.5)
        idx_fps = utils.farthest_point_sample(xyz, sample_to)
        xyz = utils.index_points(xyz, idx_fps)
        fea = utils.index_points(fea, idx_fps)

        # 更新特征
        fea = self.attention3(fea, xyz)

        # FPS取1/2的点
        sample_to = int(n_pnt * 0.5)
        idx_fps = utils.farthest_point_sample(xyz, sample_to)
        xyz = utils.index_points(xyz, idx_fps)
        fea = utils.index_points(fea, idx_fps)

        # 更新特征
        fea = self.attention4(fea, xyz)

        # 聚合特征，max pooling
        fea = torch.max(fea, dim=-2)[0]
        pred_cls = self.final_pred(fea)
        pred_cls = F.log_softmax(pred_cls, dim=-1)

        return pred_cls


if __name__ == '__main__':
    # attn = PointTransformerLayer(
    #     dim_in=128,
    #     dim_out=256,
    #     pos_mlp_hidden_dim=64,
    #     attn_mlp_hidden_mult=4,
    #     num_neighbors=16  # only the 16 nearest neighbors would be attended to for each point
    # )
    #
    # feats = torch.randn(1, 2048, 128)
    # pos = torch.randn(1, 2048, 3)
    # mask = torch.ones(1, 2048).bool()
    #
    # attn(feats, pos, mask=mask)  # (1, 16, 128)

    xyz = torch.randn(2, 2048, 3).cuda()
    classifier = AttentionCls(n_classes=6, n_chanellin=3).cuda()
    pred_classes = classifier(xyz)
    print(pred_classes.size())
