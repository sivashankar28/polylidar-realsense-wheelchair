import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
def grid(size=10, n=10, color=[0.5, 0.5, 0.5], plane='xy', plane_offset=-1, translate=[0, 0, 0]):
    """draw a grid on xz plane"""

    # lineset = o3d.geometry.LineSet()
    s = size / float(n)
    s2 = 0.5 * size
    points = []

    for i in range(0, n + 1):
        x = -s2 + i * s
        points.append([x, -s2, plane_offset])
        points.append([x, s2, plane_offset])
    for i in range(0, n + 1):
        z = -s2 + i * s
        points.append([-s2, z, plane_offset])
        points.append([s2, z, plane_offset])

    points = np.array(points)
    if plane == 'xz':
        points[:,[2,1]] = points[:,[1,2]]

    points = points + translate

    n_points = points.shape[0]
    lines = [[i, i + 1] for i in range(0, n_points -1, 2)]
    colors = [list(color)] * (n_points - 1)
    return points, lines, colors

def set_line(line_set, points, lines, colors):
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

def create_grid(size=5, n=20):
    grid_ls = o3d.geometry.LineSet()
    my_grid = grid(size=size, n=n, plane='xy', plane_offset=0, translate=[0, 0, 0])
    set_line(grid_ls, *my_grid)
    return grid_ls

def compute_transform(geom_dict:dict):
    transform = create_transform(geom_dict['center_point'], geom_dict['rotation'])
    if geom_dict.get('post_rot') is not None:
        rm = transform[:3,:3] @ geom_dict.get('post_rot')
        transform[:3, :3] = rm

    if geom_dict['parent'] is not None:
        transform = geom_dict['parent']['transform'] @ transform
    # cache the transform
    geom_dict['transform'] = transform 
    return transform


def create_cf(size, transform):
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    cf.transform(transform)
    return cf

def create_box(geom_dict):

    half_x = geom_dict['width'] / 2.0
    half_y = geom_dict['height'] / 2.0
    half_z = 0
    diff = np.array([half_x, half_y, half_z])
    # min_bound = geom_dict['center_point'] - diff
    # max_bound = geom_dict['center_point'] + diff

    geom = o3d.geometry.TriangleMesh.create_box(geom_dict['width'], geom_dict['height'], geom_dict['depth'])
    geom.compute_vertex_normals()

    geom.vertices = o3d.utility.Vector3dVector(np.asarray(geom.vertices) - diff)

    transform = compute_transform(geom_dict)
    geom.transform(transform)

    cf = create_cf(geom_dict['depth'], transform)

    geom_dict['geom'] = geom
    return [geom, cf]

def create_point(geom_dict):

    geom = o3d.geometry.TriangleMesh.create_octahedron(geom_dict['size'])
    geom.compute_vertex_normals()

    transform = compute_transform(geom_dict)
    geom.transform(transform)
    geom.paint_uniform_color(geom_dict['color'])
    cf = create_cf(geom_dict['size'], transform)
    geom_dict['geom'] = geom
    return [geom, cf]

def tranform_vectors(vectors, transform):
    point = np.array([0.0, 0.0, 0.0 , 1])
    point[:3] = vectors
    new_point = (transform @ point)[:3]
    return new_point

def rotate_vectors(vectors, transform):
    rot = transform[:3, :3]
    new_vec = rot @ vectors
    return new_vec

def create_transform(translate, rotation):
    transform = np.eye(4)
    transform[:3, 3] = translate
    if isinstance(rotation,np.ndarray):
        transform[:3, :3] = rotation
    elif isinstance(rotation, dict):
        rot = R.from_euler('xyz', angles=[rotation['pitch'], rotation['roll'], rotation['yaw']], degrees=True)
        rm = rot.as_matrix()
        transform[:3, :3] = rm
    else:
        raise ValueError("Rotation must be an ndarray or a dictionary object with euler angles")
    return transform