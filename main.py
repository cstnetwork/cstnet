import numpy as np
import pymesh
import open3d as o3d
from scipy.spatial import Delaunay
import scipy
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import copy


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。
    points = np.loadtxt(r'D:\document\DeepLearning\ParPartsNetWork\data_set_p2500_n10000\cuboid\PointCloud0.txt')

    tri = Delaunay(points)



    plt.plot(points[:, 0], points[:, 1], 'o')

    for simplex in tri.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    plt.show()


def o3dpossion():
    # 读取点云数据
    # point_cloud = np.loadtxt(r'D:\document\DeepLearning\ParPartsNetWork\data_set_p2500_n10000\cuboid\PointCloud0.txt')
    # point_cloud = o3d.io.read_point_cloud(r'C:\Users\ChengXi\Desktop\PointCloud0.txt')
    #
    # # 执行重建算法
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
    #
    # # 可选：优化 Mesh
    # mesh = mesh.filter_smooth_laplacian()
    #
    # # 可选：显示结果
    # o3d.visualization.draw_geometries([mesh])

    pcd = o3d.io.read_point_cloud(r'C:\Users\ChengXi\Desktop\PointCloud0.pcd')

    radius1 = 100  # 搜索半径
    max_nn = 100  # 邻域内用于估算法线的最大点数
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius1, max_nn))  # 执行法线估计
    # 可视化
    o3d.visualization.draw_geometries([pcd],
                                      window_name="可视化参数设置",
                                      width=600,
                                      height=450,
                                      left=30,
                                      top=30,
                                      point_show_normal=True)

    # 滚球半径的估计
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.1 * avg_dist
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     pcd,
    #     o3d.utility.DoubleVector([radius, radius * 2]))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)

    print(mesh.get_surface_area())
    o3d.visualization.draw_geometries([mesh], window_name='Open3D downSample', width=800, height=600, left=50,
                                      top=50, point_show_normal=True, mesh_show_wireframe=True,
                                      mesh_show_back_face=True, )
    # 从open3d创建具有顶点和面的三角形网格
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                               vertex_normals=np.asarray(mesh.vertex_normals))
    trimesh.convex.is_convex(tri_mesh)
    # mesh保存
    tri_mesh.export("cat_hole.ply")
    # o3d.io.write_triangle_mesh("hole.obj", tri_mesh)

    # 计算邻接关系
    adjacency_list = mesh.compute_adjacency_list()

    # 选择要查询的点的索引
    point_index = 0

    print(np.asarray(adjacency_list.triangles))

    print('type:', type(adjacency_list.triangles))








if __name__ == '__main__':
    # print_hi('PyCharm')
    o3dpossion()



