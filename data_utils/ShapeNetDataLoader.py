# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
import random
import torch

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        '''
        返回
        [0]点集 (2048, 3) <class 'numpy.ndarray'>
        [1]类别 (1,) <class 'numpy.ndarray'>
        [2]分割 (2048,) <class 'numpy.ndarray'>
        '''
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)


def shuffle20000():
    '''
    打乱顺序，可用于生成训练集测试集索引
    '''
    alist = [i for i in range(20000)]
    random.shuffle(alist)
    with open('ParaPntsTrain.txt', 'w') as f:
        for i in alist:
            f.write('{}\n'.format(i))

def MapIntToCF(fpInd, num_examples_per_class, prism_deg):
    '''
    将int映射到类型和文件名
    num_examples_per_class: 单个类别零件的点云数量（训练集 + 测试集）
    '''
    if fpInd < num_examples_per_class:
        return 'prism', 'prism{}/PointCloud{}'.format(prism_deg, fpInd) + '.txt'
    else:
        return 'cuboid', 'cuboid/PointCloud{}'.format(fpInd - num_examples_per_class) + '.txt'


class ParaPointNet2_Dataset(Dataset):
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
                 prism_deg,
                 nall_dataitem=5000, # 每类零件点云总数（训练 + 测试），用于根据编号查找对应的点云文件
                 root='.\\data\\para_pointcloud', # 数据集文件夹路径
                 npoints=1500, # 每个点云文件的点数
                 is_train=True, # 判断返回训练数据集还是测试集
                 data_augmentation=True # 是否数据增强
                 ):
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
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)

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

        point_set = torch.from_numpy(point_set)
        cls = np.array([cls]).astype(np.int32)

        return point_set.numpy(), cls, CMLabel

    def __len__(self):
        return len(self.datapath)

class SegCuboidDataset(Dataset):
    '''参数化带约束草图数据集，parametric PointNet dataset
    返回：
    ① 点
    ② 类型
    ③ cls
    读取数据：前三列为xyz，最后一列为约束
    x y z c
    x y z c
    ...
    x y z c
    '''
    def __init__(self,
                 is_train,
                 root='D:\document\DeepLearning\ParPartsNetWork\data_set\cuboid',
                 npoints=2500, # 每个点云文件的点数
                 data_augmentation=True # 是否数据增强
                 ):
        self.npoints = npoints
        self.root = root
        self.data_augmentation = data_augmentation

        self.is_train = is_train
        self.n_train = 8000
        self.n_test = 2000

    def __getitem__(self, index):  # 作用为输出点云中点的三维坐标及对应类别，类型均为tensor，类别为1×1的矩阵
        '''
        [0]点集 (2048, 3) <class 'numpy.ndarray'>
        [1]类别 (1,) <class 'numpy.ndarray'>
        [2]分割 (2048,) <class 'numpy.ndarray'>
        '''
        if(self.is_train):
            point_set = np.loadtxt(
                self.root + '\PointCloud{}.txt'.format(index))  # n*4 的矩阵,存储点数据,前三列为xyz,最后一列为约束c,------------------
        else:
            point_set = np.loadtxt(
                self.root + '\PointCloud{}.txt'.format(self.n_train + index))  # n*4 的矩阵,存储点数据,前三列为xyz,最后一列为约束c,------------------

        # 从 np.arange(len(seg))中随机选数，数量为self.npoints，replace：是否可取相同数字，replace=true表示可取相同数字，可规定每个元素的抽取概率，默认均等概率
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)

        # 从读取到的点云文件中随机取指定数量的点
        point_set = point_set[choice, :]

        # 约束矩阵的标签
        SegLabel = point_set[:, 3]

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

        point_set = torch.from_numpy(point_set)

        return point_set.numpy(), 0, SegLabel

    def __len__(self):
        if self.is_train:
            return self.n_train
        else:
            return self.n_test


