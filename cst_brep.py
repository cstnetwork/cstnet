# open cascade
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SOLID, TopAbs_ShapeEnum, TopAbs_REVERSED
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.TopoDS import topods
from OCC.Core.Precision import precision
from OCC.Core.GProp import GProp_PGProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Geom import Geom_ConicalSurface, Geom_Plane, Geom_CylindricalSurface, Geom_Curve
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.TDF import TDF_LabelSequence
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.TopTools import TopTools_ShapeMapHasher
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.BRepTools import breptools
# others
import os
import open3d as o3d
import pymeshlab
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime
import multiprocessing
import itertools

# self
import mesh_proc
import utils


class Point3DForDataSet(gp_Pnt):
    def __init__(self, pnt_loc: gp_Pnt, aligned_face: TopoDS_Face, aligned_shape_area, edges_useful):
        super().__init__(pnt_loc.XYZ())

        self.edges_useful = edges_useful

        self.aligned_face = aligned_face
        self.aligned_shape_area = aligned_shape_area

        self.eula_angle = gp_Vec(0.0, 0.0, -1.0)
        self.is_edge_nearby = 0
        self.edge_nearby_threshold = 0.0

        self.meta_eula_cal()
        self.edge_nearby_cal()

    def meta_eula_cal(self):
        aligned_surface = BRep_Tool.Surface(self.aligned_face)
        surface_type = aligned_surface.DynamicType()
        type_name = surface_type.Name()

        if type_name == 'Geom_ConicalSurface':
            self.meta_type = 1
            self.eula_angle = Geom_ConicalSurface.DownCast(aligned_surface).Axis().Direction().XYZ()
            self.eula_angle_rectify()

        elif type_name == 'Geom_CylindricalSurface':
            self.meta_type = 2
            self.eula_angle = Geom_CylindricalSurface.DownCast(aligned_surface).Axis().Direction().XYZ()
            self.eula_angle_rectify()

        elif type_name == 'Geom_Plane':
            self.meta_type = 3
            self.eula_angle = Geom_Plane.DownCast(aligned_surface).Axis().Direction().XYZ()
            self.eula_angle_rectify()

        else:
            self.meta_type = 0

    def eula_angle_rectify(self):
        ax_x = self.eula_angle.X()
        ax_y = self.eula_angle.Y()
        ax_z = self.eula_angle.Z()

        zero_lim = precision.Confusion()
        if ax_z < -zero_lim:
            self.eula_angle *= -1.0
        elif abs(ax_z) <= zero_lim:
            if ax_y < -zero_lim:
                self.eula_angle *= -1.0
            elif abs(ax_y) <= zero_lim:
                if ax_x < -zero_lim:
                    self.eula_angle *= -1.0

    def nearby_threshold_cal(self):
        rsphere = np.sqrt(self.aligned_shape_area / (4.0 * np.pi))
        # near_rate = 0.08
        near_rate = 0.03
        self.edge_nearby_threshold = near_rate * rsphere

    def is_target_edge_nearby(self, fp_edge: TopoDS_Edge):
        current_dis = dist_point2shape(self, fp_edge)
        if current_dis < self.edge_nearby_threshold:
            return True
        else:
            return False

    def edge_nearby_cal(self):
        self.nearby_threshold_cal()

        edge_explorer = TopExp_Explorer(self.aligned_face, TopAbs_EDGE)

        while edge_explorer.More():
            edge = edge_explorer.Current()
            edge = topods.Edge(edge)
            edge_explorer.Next()

            if is_edge_useful(edge, self.edges_useful) and self.is_target_edge_nearby(edge):
                self.is_edge_nearby = 1
                return

    def get_save_str(self, is_contain_xyz=True):
        if is_contain_xyz:
            save_str = (f'{self.X()}\t{self.Y()}\t{self.Z()}\t' +
                        f'{self.eula_angle.X()}\t{self.eula_angle.Y()}\t{self.eula_angle.Z()}\t' +
                        f'{self.is_edge_nearby}\t{self.meta_type}\n')
        else:
            save_str = (f'{self.eula_angle.X()}\t{self.eula_angle.Y()}\t{self.eula_angle.Z()}\t' +
                        f'{self.is_edge_nearby}\t{self.meta_type}\n')

        return save_str


def step_read_ctrl(filename):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)
    if status == IFSelect_RetDone:
        step_reader.NbRootsForTransfer()
        step_reader.TransferRoot()
        model_shape = step_reader.OneShape()

        if model_shape.IsNull():
            raise ValueError('Empty STEP file')
        else:
            return step_reader.OneShape()

    else:
        raise ValueError('Cannot read the file')


def step_read_ocaf(filename):
    _shapes = []
    cafReader = STEPCAFControl_Reader()
    aDoc = TDocStd_Document("MDTV-XCAF")

    status = cafReader.ReadFile(filename)
    if status == IFSelect_RetDone:
        cafReader.Transfer(aDoc)
    else:
        raise ValueError('STET cannot be parsed:', filename)

    rootLabel = aDoc.Main()
    ShapeTool = XCAFDoc_DocumentTool.ShapeTool(rootLabel)

    aSeq = TDF_LabelSequence()
    ShapeTool.GetFreeShapes(aSeq)

    for i in range(aSeq.Length()):
        label = aSeq.Value(i + 1)
        loc = ShapeTool.GetLocation(label)
        part = TopoDS_Shape()
        ShapeTool.GetShape(label, part)

        if not loc.IsIdentity():
            part = part.Moved(loc)

        _shapes.append(part)

    return shapes_fuse(_shapes)


def step2stl(step_name, stl_name, deflection=0.1):
    shape_occ = step_read_ocaf(step_name)
    shapeocc2stl(shape_occ, stl_name, deflection)


def step2stl_batched_(dir_path, deflection=0.1):
    step_path_all = utils.get_allfiles(dir_path, 'step')
    n_step = len(step_path_all)

    for idx, c_step in enumerate(step_path_all):
        print(f'{idx} / {n_step}')
        stl_path = os.path.splitext(c_step)[0] + '.stl'

        step2stl(c_step, stl_path, deflection)


def shapes_fuse(shapes: list):
    if len(shapes) == 0:
        return TopoDS_Shape()

    elif len(shapes) == 1:
        return shapes[0]

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    for shape in shapes:
        # if shape.ShapeType() == TopAbs_SOLID:
        builder.Add(compound, shape)

    return compound


def shape_area(shape_occ: TopoDS_Shape):
    props = GProp_PGProps()
    brepgprop.SurfaceProperties(shape_occ, props)
    return props.Mass()


def shapeocc2stl(shape_occ, save_path, deflection=0.1):
    save_path = os.path.abspath(save_path)

    mesh = BRepMesh_IncrementalMesh(shape_occ, deflection)
    mesh.Perform()
    assert mesh.IsDone()

    stl_writer = StlAPI_Writer()
    stl_writer.Write(shape_occ, save_path)


def is_edge_overlap(edge1: TopoDS_Edge, edge2: TopoDS_Edge) -> bool:
    return TopTools_ShapeMapHasher.IsEqual(edge1, edge2)


def is_edge_valid(fp_edge: TopoDS_Edge):
    curve = BRep_Tool.Curve(fp_edge)[0]

    if isinstance(curve, Geom_Curve):
        return True

    else:
        return False


def is_edge_useful(edge: TopoDS_Edge, edges_useful: TopTools_IndexedMapOfShape):
    assert edges_useful.Size() != 0

    if edges_useful.Contains(edge):
        return True
    else:
        return False


def is_point_in_shape(point: gp_Pnt, shape: TopoDS_Shape, tol: float = precision.Confusion()):
    dist2shape = dist_point2shape(point, shape)

    if dist2shape < tol:
        return True
    else:
        return False


def is_suffix_step(filename):
    if filename[-4:] == '.stp' \
            or filename[-5:] == '.step' \
            or filename[-5:] == '.STEP':
        return True

    else:
        return False


def dist_point2shape(point: gp_Pnt, shape: TopoDS_Shape):
    vert = BRepBuilderAPI_MakeVertex(point)
    vert = vert.Shape()
    extrema = BRepExtrema_DistShapeShape(vert, shape)
    extrema.Perform()
    if not extrema.IsDone() or extrema.NbSolution() == 0:
        raise ValueError('fail to compute the distance from point to shape')

    nearest_pnt = extrema.PointOnShape2(1)
    return point.Distance(nearest_pnt)


def edge_filter(edge_list: list):
    valid_edges = []

    for edge in edge_list:
        if is_edge_valid(edge):
            valid_edges.append(edge)

    return valid_edges


def get_edges_useful(shape_occ: TopoDS_Shape):
    def is_edge_in_face(fp_edge, fp_face):
        edge_exp = TopExp_Explorer(fp_face, TopAbs_EDGE)
        while edge_exp.More():
            edge_local = edge_exp.Current()
            edge_local = topods.Edge(edge_local)
            edge_exp.Next()

            if TopTools_ShapeMapHasher.IsEqual(fp_edge, edge_local):
                return True

        return False

    def find_adjfaces():
        edge_adjface = {}

        for i in range(1, edges_all.Size() + 1):
            cdege = edges_all.FindKey(i)
            adjfaces = []

            face_exp = TopExp_Explorer(shape_occ, TopAbs_FACE)
            while face_exp.More():
                if len(adjfaces) == 2:
                    break

                face_local = face_exp.Current()
                face_local = topods.Face(face_local)
                face_exp.Next()

                if is_edge_in_face(cdege, face_local):
                    adjfaces.append(face_local)

            if len(adjfaces) == 2:
                edge_adjface[cdege] = adjfaces

        return edge_adjface

    def get_face_nornal_at_pnt(fp_point: gp_Pnt, fp_face: TopoDS_Face):
        surf_local = BRep_Tool.Surface(fp_face)
        proj_local = GeomAPI_ProjectPointOnSurf(fp_point, surf_local)

        if proj_local.IsDone():
            fu, fv = proj_local.Parameters(1)
            face_props = GeomLProp_SLProps(surf_local, fu, fv, 1, precision.Confusion())
            normal_at = face_props.Normal()

            return normal_at

        else:
            raise ValueError('Can not perform projection')

    def is_edge_useful_by_adjface(fp_edge, adj_face1, adj_face2):
        acurve_info = BRep_Tool.Curve(fp_edge)
        acurve, p_start, p_end = acurve_info
        mid_pnt = acurve.Value((p_start + p_end) / 2.0)

        norm1 = get_face_nornal_at_pnt(mid_pnt, adj_face1)
        norm2 = get_face_nornal_at_pnt(mid_pnt, adj_face2)

        angle = norm1.Angle(norm2)
        prec_resolution = precision.Confusion() + 1e-5

        if angle < prec_resolution or abs(angle - np.pi) < prec_resolution:
            return False
        else:
            return True

    edges_useful = TopTools_IndexedMapOfShape()
    edges_useful.Clear()
    edges_all = TopTools_IndexedMapOfShape()

    edge_explorer = TopExp_Explorer(shape_occ, TopAbs_EDGE)
    while edge_explorer.More():
        edge = edge_explorer.Current()
        edge = topods.Edge(edge)
        edge_explorer.Next()

        try:
            if is_edge_valid(edge):
                edges_all.Add(edge)
        except:
            print('valid edge error, skip')

    edge_face_pair = find_adjfaces()

    for cedge in edge_face_pair.keys():
        try:
            if is_edge_useful_by_adjface(cedge, *edge_face_pair[cedge]):
                edges_useful.Add(cedge)
        except:
            print('valid edge error, skip')

    return edges_useful


def get_point_aligned_face(model_occ: TopoDS_Shape, point: gp_Pnt, prec=0.1):
    explorer = TopExp_Explorer(model_occ, TopAbs_FACE)

    while explorer.More():
        face = explorer.Current()
        face = topods.Face(face)
        explorer.Next()

        try:
            current_dist = dist_point2shape(point, face)
        except:
            print('can not compute point dist to face, skip current face')
            continue

        if current_dist < prec + precision.Confusion():
            return face

    return None


def get_logger(name: str = 'log'):
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log'), exist_ok=True)
    file_handler = logging.FileHandler(f'log/{name}-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def step2pcd(step_path, n_points, save_path, deflection=0.1, xyz_only=True):
    tmp_stl = 'tmp/gen_pcd_cst.stl'
    step2stl(step_path, tmp_stl)
    vertex_matrix = mesh_proc.get_points_mslab(tmp_stl, n_points)

    if xyz_only:
        np.savetxt(save_path, vertex_matrix, fmt='%.6f', delimiter='\t')

    else:
        n_points_real = vertex_matrix.shape[0]

        model_occ = step_read_ocaf(step_path)
        model_area = shape_area(model_occ)
        edges_useful = get_edges_useful(model_occ)

        if edges_useful.Size() == 0:
            print('current model without Valid Edges')

        save_path = os.path.abspath(save_path)
        with open(save_path, 'w') as file_write:
            for i in tqdm(range(n_points_real), total=n_points_real):

                current_point = gp_Pnt(float(vertex_matrix[i, 0]), float(vertex_matrix[i, 1]),
                                       float(vertex_matrix[i, 2]))

                face_aligned = get_point_aligned_face(model_occ, current_point, deflection)

                if face_aligned is not None:
                    current_datapoint = Point3DForDataSet(current_point, face_aligned, model_area, edges_useful)
                    file_write.writelines(current_datapoint.get_save_str())

                else:
                    print(
                        f'find a point({current_point.X()}, {current_point.Y()}, {current_point.Z()}) without aligned face, skip')

    os.remove(tmp_stl)


def step2pcd_faceseg(step_path, n_points, save_path, deflection=0.1):
    tmp_stl = 'tmp/gen_pcd_cst.stl'
    step2stl(step_path, tmp_stl)
    vertex_matrix = mesh_proc.get_points_mslab(tmp_stl, n_points)

    n_points_real = vertex_matrix.shape[0]

    model_occ = step_read_ocaf(step_path)

    faceidx_dict = {}
    explorer = TopExp_Explorer(model_occ, TopAbs_FACE)

    face_count = 0
    while explorer.More():
        face = explorer.Current()
        face = topods.Face(face)
        explorer.Next()

        faceidx_dict[face] = face_count
        face_count += 1

    save_path = os.path.abspath(save_path)
    with open(save_path, 'w') as file_write:
        for i in tqdm(range(n_points_real), total=n_points_real):

            current_point = gp_Pnt(float(vertex_matrix[i, 0]), float(vertex_matrix[i, 1]), float(vertex_matrix[i, 2]))

            face_aligned = get_point_aligned_face(model_occ, current_point, deflection)

            if face_aligned is not None:
                aligned_faceidx = faceidx_dict[face_aligned]
                file_write.writelines(f'{float(vertex_matrix[i, 0])}\t{float(vertex_matrix[i, 1])}\t{float(vertex_matrix[i, 2])}\t{int(aligned_faceidx)}\n')

            else:
                print(
                    f'find a point({current_point.X()}, {current_point.Y()}, {current_point.Z()}) without aligned face, skip')

    os.remove(tmp_stl)


def step2pcd_batched(dir_path, n_points=2650, is_load_progress=True, xyz_only=False, deflection=0.1):
    """
    sort to following data structure
    dir_path
    └─ raw
        ├─ car
        │   ├─ car0.stp
        │   ├─ car1.stp
        │   ├─ ...
        │   │
        │   ├─ small_car
        │   │   ├─ small_car0.stp
        │   │   ├─ small_car1.stp
        │   │   ├─ small_car2.stp
        │   │   ...
        │   │
        │   ├─ large_car
        │   │   ├─ large_car0.stp
        │   │   ├─ large_car1.stp
        │   │   ├─ large_car2.stp
        │   │   ...
        │   │
        │   ├─ car1.stp
        │   ...
        │
        ├─ plane
        │   ├─ plane0.stp
        │   ├─ plane1.stp
        │   ├─ plane2.stp
        │   ...
        │
        ...
    """

    logger = get_logger(dir_path.split(os.sep)[-1])

    pcd_dir = os.path.join(dir_path, 'pointcloud')
    os.makedirs(pcd_dir, exist_ok=True)

    path_allclasses = Path(os.path.join(dir_path, 'raw'))
    classes_all = utils.get_subdirs(path_allclasses)

    utils.create_subdirs(pcd_dir, classes_all)

    class_file_all = {}
    for curr_class in classes_all:
        curr_read_save_paths = []

        trans_count = 0
        currclass_path = os.path.join(dir_path, 'raw', curr_class)
        for root, dirs, files in os.walk(currclass_path):
            for file in files:
                current_filepath = str(os.path.join(root, file))

                if is_suffix_step(current_filepath):
                    file_name_pcd = str(trans_count) + '.txt'
                    trans_count += 1

                    current_savepath = os.path.join(dir_path, 'pointcloud', curr_class, file_name_pcd)
                    curr_read_save_paths.append((current_filepath, current_savepath))

        if len(curr_read_save_paths) != 0:
            class_file_all[curr_class] = curr_read_save_paths

    def get_progress():
        try:
            with open(filename_json, 'r') as file_json:
                progress = json.load(file_json)
        except:
            progress = {
                'dir_path': dir_path,
                'is_finished': False,
                'class_ind': 0,
                'instance_ind': 0
            }
            with open(filename_json, 'w') as file_json:
                json.dump(progress, file_json, indent=4)
        return progress

    def save_progress2json(progress_dict, class_ind, instance_ind):
        progress_dict['class_ind'] = class_ind
        progress_dict['instance_ind'] = instance_ind
        progress_dict['is_finished'] = False
        with open(filename_json, 'w') as file_json:
            json.dump(progress_dict, file_json, indent=4)

    def save_finish2json(progress_dict):
        progress_dict['is_finished'] = True
        with open(filename_json, 'w') as file_json:
            json.dump(progress_dict, file_json, indent=4)

    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
    os.makedirs(config_dir, exist_ok=True)
    filename_json = os.path.abspath(dir_path)
    filename_json = filename_json.replace(os.sep, '-').replace(':', '') + '.json'
    filename_json = os.path.join(config_dir, filename_json)

    trans_progress = get_progress()

    if trans_progress['is_finished']:
        print('no re transform' + filename_json)
        return

    startind_class = 0
    startind_instance = 0
    if is_load_progress:
        if not trans_progress['is_finished'] and trans_progress['dir_path'] == dir_path:
            startind_class = trans_progress['class_ind']
            startind_instance = trans_progress['instance_ind']
            print('load progress from json: ' + filename_json + f'- class_ind:{startind_class} - instance_ind:{startind_instance}')

    trans_count_all = 0

    class_ind = startind_class
    for curr_class in itertools.islice(class_file_all.keys(), startind_class, None):

        if class_ind == startind_class:
            instance_ind = startind_instance
            startind_instance = startind_instance
        else:
            instance_ind = 0
            startind_instance = 0

        for curr_read_save_paths in itertools.islice(class_file_all[curr_class], startind_instance, None):
            try:
                save_progress2json(trans_progress, class_ind, instance_ind)

                print('c-trans：', curr_read_save_paths[0], f'cls-file idx：{class_ind}-{instance_ind}', 'trans prog：', trans_count_all)
                print('c-save：', curr_read_save_paths[1], 'time：' + datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
                step2pcd(curr_read_save_paths[0], n_points, curr_read_save_paths[1], deflection, xyz_only)
            except:
                print('can not read c-step, skip:', curr_read_save_paths[0].encode('gbk', errors='ignore'))
                logger.info('skip:' + curr_read_save_paths[0])
                continue

            instance_ind += 1
            trans_count_all += 1
        class_ind += 1

    save_finish2json(trans_progress)


def step2pcd_batched_(dir_path, n_points=2650, xyz_only=False, deflection=0.1):
    step_path_all = utils.get_allfiles(dir_path, 'step')
    n_step = len(step_path_all)

    for idx, c_step in enumerate(step_path_all):
        print(f'{idx} / {n_step}')
        pcd_path = os.path.splitext(c_step)[0] + '.txt'
        step2pcd(c_step, n_points, pcd_path, deflection, xyz_only)


def step2pcd_multi_batched(dirs_all: list):
    threads_all = []

    for c_dir in dirs_all:
        c_thread = multiprocessing.Process(target=step2pcd_batched, args=(c_dir,))
        c_thread.start()
        threads_all.append(c_thread)

    for c_thread in threads_all:
        c_thread.join()


if __name__ == '__main__':
    step2pcd_batched(r'')
    pass






