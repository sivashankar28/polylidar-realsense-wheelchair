import argparse
from pathlib import Path
import logging
import time
from scipy.spatial.transform import Rotation as R
import warnings
from joblib import load, dump
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib import colors as mcolors
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from surfacedetector.utility.helper_general import set_axes_equal, plot_points, setup_figure_2d, setup_figure_3d
from scripts.paper.visgui import AppWindow

from surfacedetector.utility.helper_linefitting import extract_lines_wrapper_new, filter_points_from_wheel_chair, choose_plane, compute_turning_manuever, transform_points
from scripts.o3d_util import create_point

logging.basicConfig(level=logging.INFO)


warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module=r'.*scipy'
)


DATA_DIR = Path('./data/scratch_test')

font = {'family' : 'sans-serif',
        'size'   : 12}

matplotlib.rc('font', **font)

TABLEAU_COLORS = {k: mcolors.to_rgba(v) for (k,v) in mcolors.TABLEAU_COLORS.items()}

print(TABLEAU_COLORS)


ToGLCamera = np.array([
    [1,  0,  0,  0],
    [0,  -1,  0,  0],
    [0,  0,  -1,  0],
    [0,  0,  0,  1]
])
FromGLGamera = np.linalg.inv(ToGLCamera)

def model_matrix_to_extrinsic_matrix(model_matrix):
    return np.linalg.inv(model_matrix @ FromGLGamera)

def create_camera_intrinsic_from_size(width=1024, height=768, hfov=60.0, vfov=60.0):
    fx = (width / 2.0)  / np.tan(np.radians(hfov)/2)
    fy = (height / 2.0)  / np.tan(np.radians(vfov)/2)
    fx = fy # not sure why, but it looks like fx should be governed/limited by fy
    return np.array(
        [[fx, 0, width / 2.0],
         [0, fy, height / 2.0],
         [0, 0,  1]])

def save_view(vis, fname='./scripts/paper/saved_view.pkl'):
    model_matrix = np.asarray(vis.scene.camera.get_model_matrix())
    extrinsic = model_matrix_to_extrinsic_matrix(model_matrix)
    width, height = vis.size.width, vis.size.height
    intrinsic = create_camera_intrinsic_from_size(width, height)
    saved_view = dict(extrinsic=extrinsic, intrinsic=intrinsic, width=width, height=height)
    dump(saved_view, fname)

def load_view(vis, fname="./scripts/paper/saved_view.pkl"):
    try:
        saved_view = load(fname)
        vis.setup_camera(saved_view['intrinsic'], saved_view['extrinsic'], saved_view['width'], saved_view['height'])
        # Looks like the ground plane gets messed up, no idea how to fix
    except:
        print("Can't find file")

def make_line(points, lines=None, color=[0, 1, 0], add_endpoints=False):
    points = np.array(points)
    lines = np.array(lines) if lines is not None else lines_from_ordered_points(points, add_endpoints=add_endpoints)
    colors = np.array(color)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(colors)
    return ls

def lines_from_ordered_points(points, add_endpoints=False):
    lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
    if add_endpoints:
        lines.append([points.shape[0] - 1, 0])
    return np.array(lines)

def get_files():
    p = DATA_DIR.glob('*planes*')
    files = sorted([x for x in p if x.is_file()])
    return files

def make_point_cloud(pts, color=cm.tab10.colors[0][:3]):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    cloud.paint_uniform_color(color)
    return cloud

def get_camera_intrinsics_o3d(vfov=90, hfov=65, resolution=[480, 848]):
    """
    Get camera intrinsics
    """ 
    fx = resolution[0] / (2 * np.tan(hfov / 2))
    fy = resolution[1] / (2 * np.tan(vfov / 2))
    cx = resolution[0] / 2
    cy = resolution[1] / 2
    return o3d.camera.PinholeCameraIntrinsic(resolution[0], resolution[1], fx, fy, cx, cy)

def create_arrow(o3d_scene, dir_vec, position, length, material=None, label="Direction Vector", color=[1,0,0]):
    rotation_curb_normal, _ = R.align_vectors([dir_vec], [[0, 0, 1]])
    rotation_curb_normal = rotation_curb_normal.as_matrix()
    curb_normal = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.0075, cone_radius=0.02, cylinder_height=length, cone_height=0.04) \
        .rotate(rotation_curb_normal, center=[0,0,0]) \
        .translate(position) \
        .paint_uniform_color(color)
    o3d_scene.add_geometry(label, curb_normal, material)
    return curb_normal

def visualize_3d(first_points_rot, second_points_rot=None, filtered_points=None, fit_line=None, result=None,
                first_points_label="Top Plane", second_points_label="Bottom Plane", filtered_points_label="Filtered Points",
                sensor_to_wheel_chair_transform=None,
                color_image=None, depth_image=None):

    gui.Application.instance.initialize()
    use_standard_vis=True
    if use_standard_vis:
        vis = o3d.visualization.O3DVisualizer("Wheel Chair Curb Mounting Visualization", 1024, 768)
        gui.Application.instance.add_window(vis)
        vis.show_settings = True
        o3d_scene = vis
        o3d_top_scene = vis
        o3d_low_scene = vis.scene
        vis.show_ground = True
        vis.ground_plane = o3d.visualization.rendering.Scene.GroundPlane.XY
        vis.point_size = 7
        lit = None
    else:
        win = AppWindow(1024, 900)
        o3d_scene = win._scene.scene
        o3d_top_scene = win._scene
        o3d_low_scene = o3d_scene
        lit =  win.settings.material
        lit.point_size = 8

    first_points_o3d = make_line(first_points_rot)
    second_points_o3d = None if second_points_rot is None else make_line(second_points_rot)
    filtered_points_o3d = None if filtered_points is None else make_point_cloud(filtered_points, color=cm.tab10.colors[0][:3])

    
    mat_line = o3d.visualization.rendering.Material()
    mat_line.shader = "unlitLine"
    mat_line.line_width = 5  # note that this is scaled with respect to pixels

    # win.mat_line = mat_line

    if fit_line and result:
        # Add Curb
        square_point_set = fit_line['square_points']
        square_line = make_line(square_point_set, color=TABLEAU_COLORS['tab:red'][:3], add_endpoints=True)
        o3d_scene.add_geometry("Estimated Curb", square_line, mat_line)
        # Add POI Point
        poi_pos = result['platform_poi_pos_wheel_chair_frame']
        platform_poi = dict(parent=None, center_point=poi_pos, size=0.03, rotation=dict(roll=0, pitch=0, yaw=0), color=TABLEAU_COLORS['tab:blue'][:3])
        o3d_scene.add_geometry("POI", create_point(platform_poi)[0], lit)

        dist_poi = result['dist_poi']
        poi_offset = result['poi_offset']
        curb_cp = result['platform_center_pos_wheel_chair_frame']
        # Add Curb Normal
        curb_normal = result['platform_normal_wheel_chair_frame']
        curb_normal_mesh = create_arrow(o3d_scene, curb_normal, curb_cp, poi_offset - 0.05, lit, label="Curb Normal", color=TABLEAU_COLORS['tab:red'][:3])
        # Add Direction Vector to POI
        vec2 = result['vec_wheel_chair_to_poi_2D_unit']
        vec2 = np.array([vec2[0], vec2[1], 0])
        vec2_mesh  = create_arrow(o3d_scene, vec2, [0, 0, 0], dist_poi - 0.05, lit, label="Vec 2", color=TABLEAU_COLORS['tab:blue'][:3])
        # Add Inverted Curb Normal
        inv_curb_normal = -curb_normal
        inv_curb_normal_mesh  = create_arrow(o3d_scene, inv_curb_normal, [0, 0, 0], dist_poi - 0.05, lit, label="Inv Curb Normal", color=TABLEAU_COLORS['tab:purple'][:3])


    o3d_low_scene.show_ground_plane(True, o3d.visualization.rendering.Scene.GroundPlane.XY)
    o3d_scene.add_geometry(first_points_label, first_points_o3d, mat_line)
    # Create Coordinate System
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    o3d_scene.add_geometry(f"Wheel Chair Frame", cf, lit)
    if second_points_o3d is not None:
        o3d_scene.add_geometry(second_points_label, second_points_o3d, mat_line)
    if filtered_points_o3d is not None:
        o3d_scene.add_geometry(filtered_points_label, filtered_points_o3d, lit)


    if color_image is not None and not use_standard_vis:
        img_1 = o3d.geometry.Image(color_image[...,::-1].copy())
        img_2 = o3d.geometry.Image(depth_image[...,::-1].copy())
        win.rgb_widget.update_image(img_1)
        win.depth_widget.update_image(img_2)


    cam_transform = np.eye(4) if sensor_to_wheel_chair_transform is None else np.linalg.inv(sensor_to_wheel_chair_transform)
    ls_camera = o3d.geometry.LineSet.create_camera_visualization(get_camera_intrinsics_o3d(), cam_transform, scale=0.1)
    o3d_scene.add_geometry("Camera", ls_camera, mat_line)
    try:
        bounds = o3d_low_scene.bounding_box
        o3d_top_scene.setup_camera(60.0, bounds, bounds.get_center())
        # o3d_scene.scene.camera.copy_from( o3d_scene.scene.camera)
        # o3d_top_scene.setup_camera(60.0, [0.5, 2.5, 0], [0, -3, 2], [0, 0, 1])
    except Exception as e:
        o3d_top_scene.reset_camera_to_default()

    vis.add_action("Save Camera View", save_view)
    vis.add_action("Load Camera View", load_view)

    # import pdb; pdb.set_trace()
    gui.Application.instance.run()


def plot_fit_lines(ax, fit_lines, annotate=True):
    next(ax._get_lines.prop_cycler)
    for fit_line in fit_lines:
        poly1d_fn = fit_line['fn']
        if fit_line['flip_axis']:
            ax.plot(poly1d_fn(fit_line['x_points']), fit_line['x_points'], '-')
        else:
            points = fit_line['points']
            ax.plot(points[:, 0], poly1d_fn(points[:, 0]), '-')
        mean = fit_line['points'].mean(axis=0)
        if annotate:
            ax.annotate(f"RMSE={fit_line['rmse']:.3f}", (mean[0], mean[1]))

def visualize_2d(top_points_raw, top_points_2d, all_fit_lines, best_fit_lines):
    # top_points_2d, height, all_fit_lines, best_fit_lines = extract_lines_wrapper(
    #     top_points, top_normal, min_points_line)
    fig, ax = setup_figure_2d()
    plot_points(ax[0], top_points_raw)
    plot_points(ax[1], top_points_2d)
    for i in range(top_points_2d.shape[0]):
        ax[1].annotate(str(i), (top_points_2d[i, 0], top_points_2d[i, 1]))
    plot_fit_lines(ax[1], all_fit_lines, annotate=False)
    plot_fit_lines(ax[2], best_fit_lines)

    plt.show()

def process(data):
    """ Process the bottom and top planes dictionary """

    sensor_to_wheel_chair_transform = data['sensor_to_wheel_chair_transform']
    color_image= data['color_image']
    depth_image = data['depth_image']
    
    top_plane, bottom_plane = choose_plane(data['first_plane'], data['second_plane'])
    top_points, top_normal = top_plane['all_points'], top_plane['normal_ransac']
    bottom_points = bottom_plane['all_points']

    # For visualization, I am converting from sensor frame to wheel chair frame
    # Usually, I did all line extraction in sensor frame, but visualization looks better in wheel chair frame
    # Because all line extration code assumed wheel chair frame some modifications have to be made
    # such as chaning top normal to [0,0,1] manually and setting wheel_chair_direction_vec_sensor_frame to [0,1,0]
    top_points = transform_points(top_points, sensor_to_wheel_chair_transform)
    bottom_points = transform_points(bottom_points, sensor_to_wheel_chair_transform)
    top_normal = np.array([0.0, 0.0, 1.0])

    t1 = time.perf_counter()
    filtered_top_points = filter_points_from_wheel_chair(top_points)  # <100 us
    t2 = time.perf_counter()

    print("Visualize 3D Data")
    visualize_3d(top_points, bottom_points, filtered_top_points,
                sensor_to_wheel_chair_transform=sensor_to_wheel_chair_transform, 
                color_image=color_image, depth_image=depth_image)

    best_fit_lines = extract_lines_wrapper_new(
        filtered_top_points, top_normal, wheel_chair_direction_vec_sensor_frame=[0, 1, 0], debug=True)  # ~2ms

    platform_center_pos_wheel_chair_frame = best_fit_lines[0]['hplane_point'] 
    platform_normal_wheel_chair_frame = best_fit_lines[0]['hplane_normal']
    result = compute_turning_manuever(platform_center_pos_wheel_chair_frame, platform_normal_wheel_chair_frame, best_fit_lines[0],
                                        poi_offset=0.75, debug=True)
    plt.show()

    visualize_3d(top_points, bottom_points, None, fit_line=best_fit_lines[0], result=result,
                sensor_to_wheel_chair_transform=sensor_to_wheel_chair_transform, 
                color_image=color_image, depth_image=depth_image)

    t3 = time.perf_counter()
    ms1 = (t2-t1) * 1000
    ms2 = (t3-t2) * 1000
    ms = ms1 + ms2
    logging.debug(
        "Process Points - Filter and Simplify: %.2f, Extract Lines: %.2f", ms1, ms2)
    # visualize_3d(top_points, second_points_rot=best_fit_lines[0]['square_points'], line_1=best_fit_lines[0]['points_3d'])
    return ms


def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize Data')
    parser.add_argument('--file', help='Filename of planes extracted')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    if args.file is None:
        files = get_files()
        all_time = []
        for idx, f in enumerate(files):
            if idx < 0:  # 40:
                continue
            logging.info("Processing %s", f)
            data = load(f)
            ms = process(data)
            all_time.append(ms)
        logging.info("Mean Time %.2f", np.mean(all_time))   
    else:
        data = load(args.file)
        logging.info("Loading %s", args.file)
        ms = process(data)


if __name__ == "__main__":
    main()


# What do you want to see?
# Picture of the Environment
# 3D Pont Cloud of the Polygon, Both Surfaces, and sensor frame frustum
# 

