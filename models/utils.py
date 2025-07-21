import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import label_binarize


class full_connected_conv3d(nn.Module):
    def __init__(self, channels: list, bias: bool = True, drop_rate: float = 0.4):
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
        fea = embeddings
        for i in range(self.n_layers - 2):
            fc = self.linear_layers[i]
            bn = self.batch_normals[i]
            drop = self.drop_outs[i]

            fea = drop(F.relu(bn(fc(fea))))

        fea = self.outlayer(fea)

        return fea


def index_points(points, idx, is_label: bool = False):
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


def indexes_val(vals, inds):
    bs, n_item, n_vals = inds.size()

    sequence = torch.arange(bs)
    sequence_expanded = sequence.unsqueeze(1)
    sequence_3d = sequence_expanded.tile((1, n_item))
    sequence_4d = sequence_3d.unsqueeze(-1)
    batch_indices = sequence_4d.repeat(1, 1, n_vals)

    view_shape = [n_item, n_vals]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = [bs, n_item, n_vals]
    repeat_shape[1] = 1
    channel_indices = torch.arange(n_item, dtype=torch.long).view(view_shape).repeat(repeat_shape)

    return vals[batch_indices, channel_indices, inds]


def surface_knn(points_all: "(bs, n_pnts, 3)", k_near: int = 100, n_stepk = 10):
    ind_neighbor_all, all_dist = knn(points_all, n_stepk, True)

    neighbor_index_max = torch.max(all_dist, dim=-1, keepdim=True)[1]

    new_neighinds = ind_neighbor_all.clone()

    num_ita = 0
    while True:
        n_current_neighbors = new_neighinds.size()[-1]
        indexed_all = []
        for j in range(n_current_neighbors):
            indexed_all.append(index_points(ind_neighbor_all, new_neighinds[:, :, j]))
        new_neighinds = torch.cat(indexed_all, dim=-1)

        new_neighinds = torch.sort(new_neighinds, dim=-1)[0]

        duplicates = torch.zeros_like(new_neighinds)
        duplicates[:, :, 1:] = new_neighinds[:, :, 1:] == new_neighinds[:, :, :-1]

        neighbor_index_max2 = neighbor_index_max.repeat(1, 1, new_neighinds.shape[-1])
        new_neighinds[duplicates.bool()] = neighbor_index_max2[duplicates.bool()]

        dist_neighinds = indexes_val(all_dist, new_neighinds)

        sort_dist = torch.sort(dist_neighinds, dim=-1)[0]  # -> [bs, n_point, n_near]

        sort_dist_maxind = torch.max(sort_dist, dim=-1)[1]  # -> [bs, n_point]
        valid_nnear = torch.min(sort_dist_maxind).item() + 1

        is_end_loop = False
        if valid_nnear >= k_near + 1:
            valid_nnear = k_near + 1
            is_end_loop = True

        sub_neighbor_index = torch.topk(dist_neighinds, k=valid_nnear, dim=-1, largest=False)[1]  # [0] val, [1] index

        new_neighinds = indexes_val(new_neighinds, sub_neighbor_index)

        new_neighinds = new_neighinds[:, :, 1:]

        if is_end_loop:
            break

        num_ita += 1
        if num_ita > 20:
            print('max surface knn iteration count, return knn')
            return ind_neighbor_all

    return new_neighinds


def all_metric_cls(all_preds: list, all_labels: list):
    """
    计算分类评价指标：Acc.instance, Acc.class, F1-score, mAP
    :param all_preds: [item0, item1, ...], item: [bs, n_classes]
    :param all_labels: [item0, item1, ...], item: [bs, ]， 其中必须保存整形数据
    :return: Acc.instance, Acc.class, F1-score-macro, F1-score-weighted, mAP
    """
    # 将所有batch的预测和真实标签整合在一起
    all_preds = np.vstack(all_preds)  # 形状为 [n_samples, n_classes]
    all_labels = np.hstack(all_labels)  # 形状为 [n_samples]
    n_samples, n_classes = all_preds.shape

    # 确保all_labels中保存的为整形数据
    if not np.issubdtype(all_labels.dtype, np.integer):
        raise TypeError('all_labels 中保存了非整形数据')

    # ---------- 计算 Acc.Instance ----------
    pred_choice = np.argmax(all_preds, axis=1)  # -> [n_samples, ]
    correct = np.equal(pred_choice, all_labels).sum()
    acc_ins = correct / n_samples

    # ---------- 计算 Acc.class ----------
    acc_cls = []
    for class_idx in range(n_classes):
        class_mask = (all_labels == class_idx)
        if np.sum(class_mask) == 0:
            continue
        cls_acc_sig = np.mean(pred_choice[class_mask] == all_labels[class_mask])
        acc_cls.append(cls_acc_sig)
    acc_cls = np.mean(acc_cls)

    # ---------- 计算 F1-score ----------
    f1_m = f1_score(all_labels, pred_choice, average='macro')
    f1_w = f1_score(all_labels, pred_choice, average='weighted')

    # ---------- 计算 mAP ----------
    all_labels_one_hot = label_binarize(all_labels, classes=np.arange(n_classes))

    if n_classes == 2:
        all_labels_one_hot_rev = 1 - all_labels_one_hot
        all_labels_one_hot = np.concatenate([all_labels_one_hot_rev, all_labels_one_hot], axis=1)

    ap_sig = []
    # 计算单个类别的 ap
    for i in range(n_classes):
        ap = average_precision_score(all_labels_one_hot[:, i], all_preds[:, i])
        ap_sig.append(ap)

    mAP = np.mean(ap_sig)

    return acc_ins, acc_cls, f1_m, f1_w, mAP


if __name__ == '__main__':
    pass