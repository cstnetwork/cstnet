import os
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from models import utils


class CstPntDataset(Dataset):
    def __init__(self,
                 root=r'D:\document\DeepLearning\ParPartsNetWork\DataSetNew',
                 npoints=2500,
                 data_augmentation=True,
                 is_backaddattr=True
                 ):

        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.is_backaddattr = is_backaddattr

        print('STEPMillion dataset, from:' + root)
        index_file = os.path.join(root, 'index_file.txt')

        self.datapath = []
        with open(index_file, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                current_path = os.path.join(root, 'overall', line)
                self.datapath.append(current_path)

        print('instance all:', len(self.datapath))

    def __getitem__(self, index):
        fn = self.datapath[index].strip()
        point_set = np.loadtxt(fn)  # [x, y, z, ex, ey, ez, adj, pt]

        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        except:
            exit('except an error')

        point_set = point_set[choice, :]

        if self.is_backaddattr:
            eualangle = point_set[:, 3: 6]
            is_nearby = point_set[:, 6]
            meta_type = point_set[:, 7]

        point_set = point_set[:, :3]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            point_set += np.random.normal(0, 0.02, size=point_set.shape)

        if self.is_backaddattr:
            return point_set, eualangle, is_nearby, meta_type
        else:
            return point_set

    def __len__(self):
        return len(self.datapath)


class Param20KDataset(Dataset):
    def __init__(self,
                 root=r'D:\document\DeepLearning\DataSet\MCB_PointCloud\MCBPcd_A',
                 is_train=True,
                 npoints=2500,  # 每个点云文件的点数
                 data_augmentation=True,  # 是否加噪音
                 ):
        """
        定位文件的路径如下：
        root
        ├─ train
        │   ├─ Bushes
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ├─ Clamps
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ...
        │
        ├─ test
        │   ├─ Bushes
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ├─ Clamps
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ...
        │

        """

        self.npoints = npoints
        self.data_augmentation = data_augmentation

        print('MCB dataset, from:' + root)

        if is_train:
            inner_root = os.path.join(root, 'train')
        else:
            inner_root = os.path.join(root, 'test')

        category_all = utils.get_subdirs(inner_root)
        category_path = {}  # {'plane': [Path1,Path2,...], 'car': [Path1,Path2,...]}

        for c_class in category_all:
            class_root = os.path.join(inner_root, c_class)
            file_path_all = utils.get_allfiles(class_root)

            category_path[c_class] = file_path_all

        self.datapath = []
        for item in category_path:
            for fn in category_path[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(sorted(category_path), range(len(category_path))))
        print(self.classes)
        print('instance all:', len(self.datapath))

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[fn[0]]
        point_set = np.loadtxt(fn[1])  # n*6 (x, y, z, i, j, k)

        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        except:
            exit('except a error')

        point_set = point_set[choice, :]
        point_set = point_set[:, :3]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            point_set += np.random.normal(0, 0.02, size=point_set.shape)

        return point_set, cls

    def __len__(self):
        return len(self.datapath)

    def n_classes(self):
        return len(self.classes)


def vis_confusion_mat(file_name):
    array_from_file = np.loadtxt(file_name, dtype=int)

    matrix_size = array_from_file.max() + 1
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    for i in range(array_from_file.shape[1]):
        x = array_from_file[0, i]
        y = array_from_file[1, i]
        matrix[x, y] += 1

    print(matrix)

    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Counts')
    plt.title('Confusion Matrix')
    plt.xlabel('target')
    plt.ylabel('predict')
    plt.xticks(np.arange(matrix_size))
    plt.yticks(np.arange(matrix_size))
    plt.show()


def save_confusion_mat(pred_list: list, target_list: list, save_name):
    matrix_size = max(max(pred_list), max(target_list)) + 1
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    list_len = len(pred_list)
    if list_len != len(target_list):
        return

    for i in range(list_len):
        x = pred_list[i]
        y = target_list[i]
        matrix[x, y] += 1

    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Counts')
    plt.title('Confusion Matrix')
    plt.xlabel('target')
    plt.ylabel('predict')
    plt.xticks(np.arange(matrix_size))
    plt.yticks(np.arange(matrix_size))
    plt.savefig(save_name)
    plt.close()


if __name__ == '__main__':

    pass
