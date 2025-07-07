'''
vis constraint prediction
'''
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))

import torch
import torch.nn.functional as F
from datetime import datetime
import logging
import argparse
import numpy as np
import open3d as o3d
import matplotlib as plt
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import shutil
from pathlib import Path

# from models.TriFeaPred import TriFeaPred
from data_utils.ParamDataLoader import ParamDataLoader
from data_utils.ParamDataLoader import STEPMillionDataLoader
from models.cst_pred import CstPnt
from data_utils.ModelNetDataLoader import ModelNetDataLoader


def is_suffix_step(filename):
    if filename[-4:] == '.stp' \
            or filename[-5:] == '.step' \
            or filename[-5:] == '.STEP':
        return True

    else:
        return False


def get_allfiles(dir_path, suffix='txt', filename_only=False):
    filepath_all = []

    def other_judge(file_name):
        if file_name.split('.')[-1] == suffix:
            return True
        else:
            return False

    if suffix == 'stp' or suffix == 'step' or suffix == 'STEP':
        suffix_judge = is_suffix_step
    else:
        suffix_judge = other_judge

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if suffix_judge(file):
                if filename_only:
                    current_filepath = file
                else:
                    current_filepath = str(os.path.join(root, file))
                filepath_all.append(current_filepath)

    return filepath_all


class TestStepDataLoader(Dataset):
    def __init__(self,
                 root,
                 npoints=2000,
                 is_addattr=False,
                 xyz_suffix='txt'
                 ):

        self.npoints = npoints
        self.is_addattr = is_addattr

        print('pred vis dataset, from:' + root)

        pcd_all = get_allfiles(root, xyz_suffix)

        self.datapath = []

        for c_pcd in pcd_all:
            self.datapath.append(c_pcd)

        print('instance all:', len(self.datapath))

    def __getitem__(self, index):
        fn = self.datapath[index].strip()
        point_set = np.loadtxt(fn)  # [x, y, z, ex, ey, ez, near, meta]

        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        except:
            exit('except an error')

        point_set = point_set[choice, :]

        if self.is_addattr:
            euler_angle = point_set[:, 3:6]
            edge_nearby = point_set[:, 6]
            primitive_type = point_set[:, 7]

        point_set = point_set[:, :3]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.is_addattr:
            return point_set, euler_angle, edge_nearby, primitive_type, fn
        else:
            return point_set, fn

    def __len__(self):
        return len(self.datapath)


def vis_pointcloud(points, euler_angle, edge_nearby, meta_type, attr=None, show_normal=False, azimuth=45-90, elevation=45+90):
    data_all = torch.cat([points, euler_angle, edge_nearby, meta_type], dim=-1).cpu().numpy()

    def spherical_to_cartesian():
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)

        x = np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = np.sin(elevation_rad)

        return [x, y, z]

    def get_default_view():
        # v_front = [-0.30820448, 0.73437657, 0.60473222]
        # v_up = [ 0.29654273, 0.67816801, -0.67242142]

        v_front = [-0.62014676, 0.5554101, -0.55401951]
        v_up = [0.45952492, 0.82956329, 0.31727212]

        return v_front, v_up

    pcd = o3d.geometry.PointCloud()
    points = data_all[:, 0:3]
    pcd.points = o3d.utility.Vector3dVector(points)

    if show_normal:
        normals = data_all[:, 3: 6]
        normals = (normals + 1) / 2
        pcd.colors = o3d.utility.Vector3dVector(normals)

        # normals = data_all[:, 3: 6]
        # pcd.normals = o3d.utility.Vector3dVector(normals)

    if attr is not None:
        labels = data_all[:, attr]

        if attr == -1:
            num_labels = 4
            colors = np.array([plt.cm.tab10(label / num_labels) for label in labels])[:, :3]  # Using tab10 colormap

        elif attr == -2:
            colors = []
            for c_attr in labels:
                if c_attr == 0:
                    # colors.append((0, 0, 0))
                    # colors.append((255, 215, 0))
                    colors.append((189 / 255, 216 / 255, 232 / 255))
                    # colors.append((60 / 255, 84 / 255, 135 / 255))

                elif c_attr == 1:
                    # colors.append((255, 215, 0))
                    # colors.append((0, 0, 0))
                    colors.append((19 / 255, 75 / 255, 108 / 255))
                    # colors.append((230 / 255, 75 / 255, 52 / 255))

                else:
                    raise ValueError('not valid edge nearby')

            colors = np.array(colors)

        else:
            raise ValueError('not valid attr')

        pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    origin_pos = [0, 0, 0]
    view_control = vis.get_view_control()

    # set up/lookat/front vector to vis
    front = spherical_to_cartesian()

    front_param, up_param = get_default_view()
    view_control.set_front(front_param)
    view_control.set_up(up_param)
    view_control.set_lookat(origin_pos)
    # because I want get first person view, so set zoom value with 0.001, if set 0, there will be nothing on screen.
    view_control.set_zoom(3)
    vis.update_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_show_normal = show_normal

    vis.poll_events()
    vis.update_renderer()

    vis.run()
    vis.destroy_window()


def vis_stl_view(stl_path):
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0., 1., 1.])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)

    origin_pos = [0, 0, 0]
    view_control = vis.get_view_control()

    v_front, v_up = [-0.62014676, 0.5554101, -0.55401951], [0.45952492, 0.82956329, 0.31727212]
    view_control.set_front(v_front)
    view_control.set_up(v_up)
    view_control.set_lookat(origin_pos)
    view_control.set_zoom(3)

    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.run()

    # camera_params = view_control.convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters("camera_params.json", camera_params)

    vis.destroy_window()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=2500, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--n_metatype', type=int, default=4, help='number of considered meta type')
    parser.add_argument('--workers', type=int, default=10, help='dataloader workers')

    parser.add_argument('--pcd_suffix', type=str, default='txt', help='-')
    parser.add_argument('--has_addattr', type=str, default='False', choices=['True', 'False'], help='-')
    parser.add_argument('--pred_model', type=str, default=r'TriFeaPred_ValidOrig_fuse', help='-')
    parser.add_argument('--root_dataset', type=str, default=r'D:\document\DeepLearning\paper_draw\AttrVis_MCB', help='root of dataset')
    # STEPMillion: r'D:\document\DeepLearning\paper_draw\AttrVis_ABC'
    # MCB: r'D:\document\DeepLearning\paper_draw\AttrVis_MCB'
    # 360Gallery: r'D:\document\DeepLearning\DataSet\test\360Gallery'

    args = parser.parse_args()
    print(args)
    return args


def main(args):
    if args.has_addattr == 'True':
        has_addattr = True
    else:
        has_addattr = False

    save_str = args.pred_model

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # eval_dataset = ModelNetDataLoader(root=args.root_dataset, args=args, split='train')
    # # eval_dataset = STEPMillionDataLoader(root=args.root_dataset, npoints=args.num_point, data_augmentation=False, is_backaddattr=False)
    # eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))  # , drop_last=True

    eval_dataset = TestStepDataLoader(root=args.root_dataset, npoints=args.num_point, is_addattr=has_addattr, xyz_suffix=args.pcd_suffix)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))  # , drop_last=True

    '''MODEL LOADING'''
    predictor = CstPnt(n_points_all=args.num_point, n_primitive=args.n_metatype).cuda()

    model_savepth = 'model_trained/' + save_str + '.pth'
    try:
        predictor.load_state_dict(torch.load(model_savepth))
        print('loading predictor from: ' + model_savepth)
    except:
        print('no existing model')
        exit(0)

    if not args.use_cpu:
        predictor = predictor.cuda()

    predictor = predictor.eval()

    with torch.no_grad():
        for batch_id, data in enumerate(eval_dataloader, 0):
            if isinstance(data, list):
                xyz = data[0]
                pcd_bs = data[-1]

                if has_addattr:
                    gt_euler = data[1].cuda()
                    gt_edge = data[2].to(torch.long).cuda()
                    gt_meta = data[3].to(torch.long).cuda()
                    gt_edge = gt_edge.unsqueeze(2)
                    gt_meta = gt_meta.unsqueeze(2)

            else:
                xyz = data

            bs = xyz.size()[0]

            xyz = xyz.float().cuda()

            pred_eula_angle, pred_edge_nearby, pred_meta_type = predictor(xyz)

            nearby = pred_edge_nearby.data.max(2)[1].unsqueeze(2)
            meta = pred_meta_type.data.max(2)[1].unsqueeze(2)

            for c_bs in range(bs):
                c_xyz = xyz[c_bs, :, :]

                vis_pointcloud(c_xyz, pred_eula_angle[c_bs, :, :], nearby[c_bs, :, :], meta[c_bs, :, :], -2)
                vis_pointcloud(c_xyz, pred_eula_angle[c_bs, :, :], nearby[c_bs, :, :], meta[c_bs, :, :], -1)
                vis_pointcloud(c_xyz, pred_eula_angle[c_bs, :, :], nearby[c_bs, :, :], meta[c_bs, :, :], None, True)

                if has_addattr:
                    vis_pointcloud(c_xyz, gt_euler[c_bs, :, :], gt_edge[c_bs, :, :], gt_meta[c_bs, :, :], -2)
                    vis_pointcloud(c_xyz, gt_euler[c_bs, :, :], gt_edge[c_bs, :, :], gt_meta[c_bs, :, :], -1)
                    vis_pointcloud(c_xyz, gt_euler[c_bs, :, :], gt_edge[c_bs, :, :], gt_meta[c_bs, :, :], None, True)

                    pcd_file = pcd_bs[c_bs]
                    stl_file = os.path.splitext(pcd_file)[0] + '.stl'
                    vis_stl_view(stl_file)

                else:
                    pcd_file = pcd_bs[c_bs]
                    stl_file = os.path.splitext(pcd_file)[0] + '.obj'
                    stl_file = str(stl_file).replace('MCB_PointCloud', 'MCB')
                    vis_stl_view(stl_file)


def remove_round_black(dir_path, pix_width=1):
    pictures_all = get_allfiles(dir_path, 'png', False)

    for c_pic in tqdm(pictures_all, total=len(pictures_all)):
        img = Image.open(c_pic)
        width, height = img.size
        crop_area = (pix_width, pix_width, width - pix_width, height - pix_width)
        cropped_img = img.crop(crop_area)
        cropped_img.save(c_pic)
        del_white_pixel(c_pic)


def del_white_pixel(png_file):
    image = Image.open(png_file).convert("RGBA")
    data = image.getdata()
    new_data = []

    for item in data:
        if item[:3] == (255, 255, 255):
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)

    image.putdata(new_data)
    image.save(png_file, "PNG")


def prepare_for_360Gallery_test(xyz_path, target_dir):
    xyz_all = get_allfiles(xyz_path, 'xyz')
    parent_folder = os.path.dirname(xyz_path)

    for idx, c_xyz in enumerate(xyz_all):
        print(f'{idx}/{len(xyz_all)}')

        file_name = os.path.splitext(os.path.basename(c_xyz))[0]
        target_xyz = os.path.join(target_dir, file_name + '.xyz')
        shutil.copy(c_xyz, target_xyz)

        source_obj = os.path.join(parent_folder, 'meshes', file_name + '.obj')
        target_obj = os.path.join(target_dir, file_name + '.obj')
        shutil.copy(source_obj, target_obj)


def get_subdirs(dir_path):
    path_allclasses = Path(dir_path)
    directories = [str(x) for x in path_allclasses.iterdir() if x.is_dir()]
    dir_names = [item.split(os.sep)[-1] for item in directories]

    return dir_names


def prepare_for_MCB_test(xyz_path, target_dir, class_ins_count=5, start_idx=0):
    classes_all = get_subdirs(xyz_path)

    for c_class in classes_all:
        if c_class != 'gear':
            continue

        print('process class:', c_class)

        c_class_dir = os.path.join(xyz_path, c_class)

        c_class_files = get_allfiles(c_class_dir)

        if len(c_class_files) < class_ins_count:
            continue

        for i in range(start_idx, start_idx + class_ins_count):
            c_xyz_source = c_class_files[i]

            file_name = os.path.splitext(os.path.basename(c_xyz_source))[0]
            c_xyz_target = os.path.join(target_dir, file_name + '.txt')

            shutil.copy(c_xyz_source, c_xyz_target)

            c_obj_source = c_xyz_source.replace('MCB_PointCloud', 'MCB')
            c_obj_source = os.path.splitext(c_obj_source)[0] + '.obj'
            c_obj_target = os.path.splitext(c_xyz_target)[0] + '.obj'

            shutil.copy(c_obj_source, c_obj_target)


if __name__ == '__main__':

    main(parse_args())

    # prepare_for_MCB_test(r'D:\document\DeepLearning\DataSet\MCB_PointCloud\MCB_A\train', r'D:\document\DeepLearning\paper_draw\AttrVis_MCB', 30)
    # prepare_for_MCB_test(r'D:\document\DeepLearning\DataSet\MCB_PointCloud\MCB_B\train', r'D:\document\DeepLearning\paper_draw\AttrVis_MCB', 100)

    # vis_stl_view(r'D:\document\DeepLearning\DataSet\STEP9000\STEP9000GenHammersleyXYZ-AddAttr\bearing\trans1\QJS200.stl')

    # asas = np.loadtxt(r'D:\document\DeepLearning\DataSet\STEPMillion\STEPMillion_pack1\overall\2629.txt')
    # print(asas)

    # remove_round_black(r'C:\Users\ChengXi\Desktop\Pred3Attr-MCB')

    # prepare_for_360Gallery_test(r'D:\document\DeepLearning\DataSet\360Gallery_Seg\point_clouds', r'D:\document\DeepLearning\DataSet\test\360Gallery')


