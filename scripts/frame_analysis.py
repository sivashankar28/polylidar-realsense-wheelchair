import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from surfacedetector.utility.line_mesh import LineMesh

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
    transform = np.eye(4)
    transform[:3, 3] = geom_dict['center_point']
    rot = geom_dict['rotation']
    rot_obj = R.from_euler('xyz', angles=[rot['pitch'], rot['roll'], rot['yaw']], degrees=True)
    rm = rot_obj.as_matrix()
    if geom_dict.get('post_rot') is not None:
        rm = rm @ geom_dict.get('post_rot')
    transform[:3, :3] = rm


    if geom_dict['parent'] is not None:
        transform = geom_dict['parent']['transform'] @ transform

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

MOUNT_TO_SENSOR_ROT = np.linalg.inv(np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
]))


def main():
    POI_OFFSET = 0.5
    PLATFORM_WIDTH = 1.5
    PLATFORM_HEIGHT = 0.25
    WHEEL_CHAIR_POS = [-1.5, -2.5, 0]

    global_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    grid = create_grid(size=5, n=20) # each square is 25X25 CM, size is 5X5

    # Platform Box, Platform Center Point, and Platform Point of Interest (POI) 
    platform = dict(parent=None, center_point=np.array([0, 2, 0]), depth=PLATFORM_HEIGHT, width=PLATFORM_WIDTH, height=1.0, rotation=dict(roll=0, pitch=0, yaw=0))
    platform_cp = dict(parent=platform, center_point=np.array([0, -0.5, 0.125]), size=0.02, rotation=dict(roll=0, pitch=0, yaw=0), color=[1,0,0])
    platform_poi = dict(parent=platform, center_point=np.array([0, -0.5 - POI_OFFSET, 0.125]), size=0.02, rotation=dict(roll=0, pitch=0, yaw=0), color=[0,1,0])

    # Wheel Chair Box, Sensor Mount, and Sensor
    wheel_chair = dict(parent=None, center_point=np.array(WHEEL_CHAIR_POS), depth=0.7, width=0.5, height=0.5, rotation=dict(roll=0, pitch=0, yaw=-45))
                                            # offset from wheel chair center          rotation from wheel chair frame
    sensor_mount = dict(parent=wheel_chair, center_point=[0.25, 0.25, 0.7], rotation=dict(roll=0, pitch=-15, yaw=0), width=0.1, depth=0.05, height=0.03)
    sensor = dict(parent=sensor_mount, center_point=[0, 0.025, 0.025], rotation=dict(roll=0, pitch=0, yaw=0), post_rot=MOUNT_TO_SENSOR_ROT, width=0.01, depth=0.025, height=0.01)

    # Create Open 3D Geometries
    platform_geom = create_box(platform)
    platform_cp_geom = create_point(platform_cp)
    platform_poi_geom = create_point(platform_poi)
    wheel_chair_geom = create_box(wheel_chair)
    sensor_mount_geom = create_box(sensor_mount)
    sensor_geom = create_box(sensor)

    # Get sensor, poi, and wheelchair 3D position and 2D projections in world frame
    sensor_pos_world = sensor['geom'].get_center()
    platform_poi_pos_world = platform_poi['geom'].get_center()
    wheel_chair_pos_world = wheel_chair['geom'].get_center()
    sensor_pos_world_proj = np.copy(sensor_pos_world)
    platform_poi_pos_world_proj = np.copy(platform_poi_pos_world)
    wheel_chair_pos_world_proj = np.copy(wheel_chair_pos_world)
    sensor_pos_world_proj[2] = 0
    platform_poi_pos_world_proj[2] = 0
    wheel_chair_pos_world_proj[2] = 0

    vec3_world = platform_poi_pos_world - sensor_pos_world
    vec3_geom_1 = LineMesh([sensor_pos_world, platform_poi_pos_world, sensor_pos_world_proj, platform_poi_pos_world_proj ], lines=[[0, 1], [2,3]], radius=0.005, colors=[1, 0, 0])
    vec3_geom_2 = LineMesh([wheel_chair_pos_world_proj, platform_poi_pos_world_proj], lines=[[0, 1]], radius=0.005, colors=[0, 0, 1])
    print(vec3_world)

    o3d.visualization.draw_geometries([global_frame, grid, *wheel_chair_geom, *platform_geom, *platform_cp_geom, 
                                        *platform_poi_geom,*sensor_mount_geom, sensor_geom[1],
                                        *vec3_geom_1.cylinder_segments, *vec3_geom_2.cylinder_segments])

if __name__ == "__main__":
    main()



