"""
为参数化点云论文绘图的脚本
"""
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
    ax.scatter(point_center[0], point_center[1], point_center[2], color=color_center, alpha=1, label='Center Points', s=25)

    # 显示邻近点
    ax.scatter(points_neighbor[:, 0], points_neighbor[:, 1], points_neighbor[:, 2], color=color_neighbor, label='Near Points', alpha=1, s=15)

    # 显示所有点
    ax.scatter(points_all[:, 0], points_all[:, 1], points_all[:, 2], color=color_other, label='Other Points', s=5)

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


def surf_knn_pral(n_neighbor = 90, name_txt = '3.txt'):
    """
    用于显示 surface knn 的原理
    :return:
    """
    file_path = r'C:\Users\ChengXi\Desktop\hardreads\cuboid_view.txt'
    # center_ind = 512
    # center_ind = 200
    center_ind = 800

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

    np.savetxt(name_txt, center_and_neighbor_ind.numpy(), fmt='%d')

    # show_neighbor_points(points_all, neighbor_points, center_pnt)


def generate4data_auto(neighbor_list=[20, 50, 80, 160]):
    for ind, n_neighbor in enumerate(neighbor_list, 1):
        surf_knn_pral(n_neighbor, str(ind)+'.txt')


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
    ax.scatter(exist_point[0], exist_point[1], exist_point[2], color=color_exist, alpha=1, label='Center Points', s=25)

    # 显示中心点
    ax.scatter(center_points[:, 0], center_points[:, 1], center_points[:, 2], color=color_center, alpha=1, label='Center Points', s=25)

    # 显示邻近点
    ax.scatter(neighbor_points[:, 0], neighbor_points[:, 1], neighbor_points[:, 2], color=color_neighbor, label='Near Points', alpha=1, s=15)

    # 显示所有点
    ax.scatter(points_all[:, 0], points_all[:, 1], points_all[:, 2], color=color_other, label='Other Points', s=5)

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
    ax.scatter(exist_point[:, 0], exist_point[:, 1], exist_point[:, 2], color=color_exist, alpha=1, label='Center Points', s=25)

    # 显示中心点
    ax.scatter(center_points[:, 0], center_points[:, 1], center_points[:, 2], color=color_center, alpha=1, label='Center Points', s=25)

    # 显示邻近点
    ax.scatter(neighbor_points[:, 0], neighbor_points[:, 1], neighbor_points[:, 2], color=color_neighbor, label='Near Points', alpha=1, s=15)

    # 显示所有点
    ax.scatter(points_all[:, 0], points_all[:, 1], points_all[:, 2], color=color_other, label='Other Points', s=5)

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
    ax.scatter(exist_point[:, 0], exist_point[:, 1], exist_point[:, 2], color=color_exist, alpha=1, label='Center Points', s=25)

    # 显示中心点
    ax.scatter(center_points[:, 0], center_points[:, 1], center_points[:, 2], color=color_center, alpha=1, label='Center Points', s=25)

    # 显示邻近点
    ax.scatter(neighbor_points[:, 0], neighbor_points[:, 1], neighbor_points[:, 2], color=color_neighbor, label='Near Points', alpha=1, s=15)

    # 显示所有点
    ax.scatter(points_all[:, 0], points_all[:, 1], points_all[:, 2], color=color_other, label='Other Points', s=5)

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


def array_subtraction(array_large, array_small):

    # 将数组转换为集合并进行差集运算
    set1 = set(array_small)
    set2 = set(array_large)
    result_set = set2 - set1

    # 将结果集合转换回Numpy数组
    result = np.array(list(result_set))

    return result


if __name__ == '__main__':

    # color_exist = [156/255, 64/255, 132/255]
    # color_center = [0, 0, 1]
    # color_neighbor = [1, 0, 0]
    # color_other = [110/255, 189/255, 183/255]

    color_exist = [63/255, 129/255, 180/255]
    color_center = [154/255, 64/255, 132/255]
    color_neighbor = [241/255, 150/255, 78/255]
    color_other = [110/255, 189/255, 183/255]

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


    # generate4data_auto([20, 50, 100, 160])
    #
    show_surfknn_paper1()
    show_surfknn_paper2()
    show_surfknn_paper3()
    show_surfknn_paper4()

    # teat_star()
    # test_where()
    # test_surfknn_testv2()
    # test_batch_indexes()
    # patch_interpolate()
    # test_unique()
    # test_knn2()


