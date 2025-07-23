import os
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from models import utils


class CstPntDataset(Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 data_augmentation=True
                 ):
        print('CstPnt dataset, from:' + root)

        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.datapath = utils.get_allfiles(root)

        print('instance all:', len(self.datapath))

    def __getitem__(self, index):
        fn = self.datapath[index].strip()
        point_set = np.loadtxt(fn)  # [x, y, z, ex, ey, ez, adj, pt]

        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        except:
            exit(f'insufficient point number of the point cloud: all points: {point_set.shape[0]}, required points: {self.npoints}')

        point_set = point_set[choice, :]

        xyz = point_set[:, :3]
        mad = point_set[:, 3: 6]
        adj = point_set[:, 6]
        pt = point_set[:, 7]

        xyz = xyz - np.expand_dims(np.mean(xyz, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)), 0)
        xyz = xyz / dist  # scale

        if self.data_augmentation:
            xyz += np.random.normal(0, 0.02, size=xyz.shape)

        return xyz, mad, adj, pt

    def __len__(self):
        return len(self.datapath)


class Param20KDataset(Dataset):
    def __init__(self,
                 root,
                 is_train=True,
                 npoints=2500,
                 data_augmentation=True,
                 is_backcst=True
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
        print('Param20K dataset, from:' + root)

        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.is_backcst = is_backcst

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
        point_set = np.loadtxt(fn[1])  # (x, y, z, mad, adj, pt)

        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        except:
            exit(f'insufficient point number of the point cloud: all points: {point_set.shape[0]}, required points: {self.npoints}')

        point_set = point_set[choice, :]
        xyz = point_set[:, :3]

        # scale points to [-1, 1]^2
        xyz = xyz - np.expand_dims(np.mean(xyz, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)), 0)
        xyz = xyz / dist

        if self.data_augmentation:
            xyz += np.random.normal(0, 0.02, size=xyz.shape)

        if self.is_backcst:
            mad = point_set[:, 3: 6]
            adj = point_set[:, 6]
            pt = point_set[:, 7]

            return xyz, cls, mad, adj, pt
        else:
            return xyz, cls

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
