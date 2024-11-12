import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
import time


def vis_pointcloudattr(pointcloud, attr):
    '''
    可视化带属性的点
    :param pointcloud: 点云 [n, 3] numpy ndarray
    :param attr: [n, ] numpy ndarray 按每行最大数值的位置可视化
    '''
    # max_indices = np.argmax(attr, axis=1).astype(np.float32)
    unique_labels = np.unique(attr)
    num_labels = len(unique_labels)
    colors = np.array([plt.cm.tab10(label / num_labels) for label in attr])[:, :3]  # Using tab10 colormap

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def visualize_point_cloud(filepath=r'D:\document\DeepLearning\DataSet\STEPMillion\pointcloud\overall\0.txt'):

    points = np.loadtxt(filepath)
    labels = points[:, -1]
    points = points[:, 0:3]

    # Generate a color map for the labels
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    print(num_labels)
    colors = np.array([plt.cm.tab10(label / num_labels) for label in labels])[:, :3]  # Using tab10 colormap

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window()
    # visualizer.add_geometry(pcd)
    # visualizer.capture_screen_image("screenshot.png")
    # visualizer.destroy_window()

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


def visualize_point_cloud2():
    # Load the point cloud data
    points = np.loadtxt('E:\document\DeepLearning\Pointnet_plus_plus\points.txt')
    labels = np.loadtxt('E:\document\DeepLearning\Pointnet_plus_plus\labels.txt')

    labels = np.argmax(labels, axis=1)

    # Load the labels
    # labels = np.loadtxt(seg_file, dtype=np.int32)
    # labels = points[:, -1]
    # points = points[:, 0:3]

    # Generate a color map for the labels
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    colors = np.array([plt.cm.tab10(label / num_labels) for label in labels])[:, :3]  # Using tab10 colormap

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window()
    # visualizer.add_geometry(pcd)
    # visualizer.capture_screen_image("screenshot.png")
    # visualizer.destroy_window()

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


def vis(points_path, pred_path):
    points = np.loadtxt(points_path)
    labels = np.loadtxt(pred_path)

    labels = np.argmax(labels, axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels)
    plt.savefig('vis.png')
    # plt.show()

    # # _, predicted_classes = pred.max(dim=1) # 获取预测的类别
    # # 类别对应的颜色
    # colors = ['b', 'g', 'r', 'c', 'm', 'y']

    # # 创建一个新的绘图
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # 绘制点云，每个类别不同颜色
    # for i in range(6):
    #     # 选择属于当前类别的点
    #     indices = labels == i
    #     # 绘制点
    #     ax.scatter(points[indices, 0], points[indices, 1], points[indices, 2], c=colors[i], label=f'Class {i}')

    #     # 添加图例
    #     ax.legend()

    #     plt.savefig('vis.png')
    #     plt.close()


def vis_bin():
    file_name = 'E:\\document\\DeepLearning\\PointPillars\\demo\data\\kitti\\000008.bin'

    # 读取点云数据
    point_cloud = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)

    # 创建 PointCloud 对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])  # 提取点的坐标信息

    # 设置点云颜色为灰色
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    # Replace 'path_to_a.pts' and 'path_to_b.seg' with the actual file paths
    # visualize_point_cloud()
    # vis('E:\document\DeepLearning\Pointnet_plus_plus\points.txt', 'E:\document\DeepLearning\Pointnet_plus_plus\labels.txt')
    # vis_bin()
    # test_tqdm()

    all_points = np.loadtxt(r'D:\document\DeepLearning\ParPartsNetWork\dataset_xindi\pointcloud\key\0.txt')
    points = all_points[:, : 3]
    attr = all_points[:, 6]

    vis_pointcloudattr(points, attr)

