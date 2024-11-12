'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    '''将pc中心平移到原点，并归一化点云规模'''
    # pc.shape = 1024,3
    # np.mean(pc, axis=0): ,第0维度的长度将变为1，[mean(pc[:, 0]), mean(pc[:, 1]), mean(pc[:, 2])]
    # 即 centroid 为点云的中心
    centroid = np.mean(pc, axis=0)

    # 将点云平移到原点
    pc = pc - centroid

    # 计算点云到原点距离的最大值
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))

    # 归一化点云规模
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    '''分类时使用的数据集加载类'''
    def __init__(self, root, args, split='train', process_data=False): # split输入：train
        self.root = root # 数据集根目录，输入：'data/modelnet40_normal_resampled/'
        self.npoints = args.num_point # 每个点云文件中的点数量，输入：1024
        self.process_data = process_data # 输入：False
        self.uniform = args.use_uniform_sample # 输入：False
        self.use_normals = args.use_normals # 输入：False
        self.num_category = args.num_category # 输入：40

        # 读取catfile的路径，catfile记录了物体类型名，每行记录一个物体名
        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        # self.cat：一个数组，每个元素为字符串，每个字符串为每类模型名['airplane', 'bathtub', 'bed', ...]
        self.cat = [line.rstrip() for line in open(self.catfile)]

        # self.classes：一个字典，键：物体名字符串，值：整形数字。{'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, ...}
        # 即用整形数字代表物体类别
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        # shape_ids：键为字符串 'train' or 'test'，值为包含了文件名字符串，类似 keyboard_0104、guitar_0128、...
        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        # shape_names：数组，去掉shape_ids['train']中字符串末尾数字
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # shape_names = [str(x.split('_')[0:-1]) for x in shape_ids[split]]

        # self.datapath：数组中每个元素为一个元组，每个元组[0]为零件名字符串，[1]为对应点云文件的绝对路径
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        # self.uniform = False。self.num_category = 40,  split = 'train',  self.npoints = 1024
        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        # self.process_data = False，是否保存数据集到本地
        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        # self.process_data = False
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            # fn：(‘物体类型名str’，‘该类型点云文件绝对路径str’)
            fn = self.datapath[index]

            # self.classes：{'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, ...}
            cls = self.classes[fn[0]]

            # 将数字cls转化为numpy数组，17 -> [17]
            label = np.array([cls]).astype(np.int32)

            # 从txt文件读取数据，delimiter：分隔符，默认空格
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            # self.uniform=False；self.npoints=1024
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # self.use_normals=False，此时截取点云的前三列，即xyz
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        # 返回一个元组，n×3/6 的 numpy 数组和 numpy.int32 类型的数字
        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


def MapIntToCF(fpInd, num_examples_per_class, prism_deg = 50):
    '''
    将int映射到类型和文件名
    num_examples_per_class: 单个类别零件的点云数量（训练集 + 测试集）
    '''
    if fpInd < num_examples_per_class:
        return 'prism', 'prism{}/PointCloud{}'.format(prism_deg, fpInd) + '.txt'
    else:
        return 'cuboid', 'cuboid/PointCloud{}'.format(fpInd - num_examples_per_class) + '.txt'

class ParaPartDataLoader(Dataset):
    '''参数化带约束草图数据集，parametric PointNet dataset
    返回：
    ① 点
    ② 类型
    ③ 约束矩阵
    读取数据：前三列为xyz，最后一列为约束
    x y z c
    x y z c
    ...
    x y z c
    '''
    def __init__(self,
                 nall_dataitem=5000, # 每类零件点云总数（训练 + 测试），用于根据编号查找对应的点云文件
                 root='./data/para_pointcloud', # 数据集文件夹路径
                 npoints=1500, # 每个点云文件的点数
                 is_train=True, # 判断返回训练数据集还是测试集
                 data_augmentation=True, # 是否数据增强
                 prism_deg=50): # 棱柱倾角
        self.npoints = npoints
        self.root = root
        self.data_augmentation = data_augmentation
        self.nall_item = nall_dataitem
        self.is_train = is_train

        datafile_ind = 'ParaPntsTest_20Percentage_Single' + str(nall_dataitem) + '.txt' # 记录数据索引的文本文件

        if(is_train):
            datafile_ind = 'ParaPntsTrain_80Percentage_Single' + str(nall_dataitem) + '.txt'

        self.catfile = os.path.join(self.root, datafile_ind)
        print('index file path: ', self.catfile)

        self.cat = {}  # {'prism': [(PCPath1,CMPath1),...], 'cuboid': [(PCPath2,CMPath2),...]}

        with open(self.catfile, 'r', encoding = "utf-8") as f:
            for line in f:
                FileInd = (int)(line.strip())
                self.cat[MapIntToCF(FileInd, self.nall_item, prism_deg)[0]] = [] # 创建每类的键及对应的值的类型，字典的键为字符串‘cuboid','prism'

        with open(self.catfile, 'r', encoding = "utf-8") as f:
            for line in f:
                FileInd = (int)(line.strip())
                self.cat[MapIntToCF(FileInd, self.nall_item, prism_deg)[0]].append(MapIntToCF(FileInd, self.nall_item, prism_deg)[1]) # 为键对应的值(数组)填充内容,内容为对应的点云在root后路径,包含文件名

        self.datapath = []  # [(‘cuboid’, APCPath), (‘prism’, APCPath), ...]存储点云的绝对路径，此外还有类型，放入同一个数组。类型，点云构成数组中的一个元素
        for item in self.cat: # item 为字典的键，即类型‘cuboid','prism'
            for fn in self.cat[item]: # fn 为每类点云对应的文件路径
                self.datapath.append((item, os.path.join(self.root, fn))) # item：类型（‘cuboid','prism'），os.path.join(self.root, fn)点云文件绝对路径

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))  # 用整形0,1,2,3等代表具体类型‘cuboid','prism'等，此时字典self.cat中的键值没有用到，self.classes的键为‘cuboid'或'prism'，值为0,1

    def __getitem__(self, index):  # 作用为输出点云中点的三维坐标及对应类别，类型均为tensor，类别为1×1的矩阵
        '''
        [0]点集 (2048, 3) <class 'numpy.ndarray'>
        [1]类别 (1,) <class 'numpy.ndarray'>
        [2]分割 (2048,) <class 'numpy.ndarray'>
        '''
        fn = self.datapath[index] # ('prism', APCPath). fn:一维元组，fn[0]：‘cuboid'或者'prism'，fn[1]：对应的点云文件路径
        cls = self.classes[fn[0]] # 表示类别的整形数字。 self.classes：键：‘cuboid'或者'prism'，值：用于表示其类型的整形数字 0,1

        point_set = np.loadtxt(fn[1]) # n*4 的矩阵,存储点数据,前三列为xyz,最后一列为约束c,------------------

        # 从 np.arange(len(seg))中随机选数，数量为self.npoints，replace：是否可取相同数字，replace=true表示可取相同数字，可规定每个元素的抽取概率，默认均等概率
        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        except:
            print('捕获异常时的point_set.shape[0]：', point_set.shape[0])
            print('捕获出错时的文件：', fn[0], '------', fn[1])
            exit('except a error')

        # 从读取到的点云文件中随机取指定数量的点
        point_set = point_set[choice, :]

        # 约束矩阵的标签
        CMLabel = point_set[:, 3]

        # 去掉点的第一列序号，切片操作左闭右开
        point_set = point_set[:, 0:3]

        # 先减去平均点，再除最大距离，即进行位置和大小的归一化
        # center # np.mean() 返回一个1*x的矩阵，axis=0 求每列的均值，axis=1，求每行的均值
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0) # 实测是否加expand_dims效果一样
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation: # 随机旋转及加上正态分布噪音
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation # 仅仅是x，y分量作旋转-----------
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter # 所有分量加正态分布随机数

        # point_set = torch.from_numpy(point_set)
        # cls = np.array([cls]).astype(np.int32)

        return point_set.astype(float), cls

    def __len__(self):
        return len(self.datapath)







if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
