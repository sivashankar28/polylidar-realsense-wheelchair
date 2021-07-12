# Author: Jeremy Castagno
# License: MIT
# The most brutal hack in order to save the view using the Python API for O3DVisualizer Class.
# I'm sure there will be a better way in the future


import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from pickle import load, dump


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

def save_view(vis, fname='saved_view.pkl'):
    try:
        model_matrix = np.asarray(vis.scene.camera.get_model_matrix())
        extrinsic = model_matrix_to_extrinsic_matrix(model_matrix)
        width, height = vis.size.width, vis.size.height
        intrinsic = create_camera_intrinsic_from_size(width, height)
        saved_view = dict(extrinsic=extrinsic, intrinsic=intrinsic, width=width, height=height)
        with open(fname, 'wb') as pickle_file:
            dump(saved_view, pickle_file)
    except Exception as e:
        print(e)

def load_view(vis, fname="saved_view.pkl"):
    try:
        with open(fname, 'rb') as pickle_file:
            saved_view = load(pickle_file)
        vis.setup_camera(saved_view['intrinsic'], saved_view['extrinsic'], saved_view['width'], saved_view['height'])
        # Looks like the ground plane gets messed up, no idea how to fix
    except Exception as e:
        print("Can't find file", e)

def main():
    gui.Application.instance.initialize()
    vis = o3d.visualization.O3DVisualizer("Demo to Load a Camera Viewpoint for O3DVisualizer", 1024, 768)
    gui.Application.instance.add_window(vis)
    vis.point_size = 8
    vis.show_axes = True
    # Add saving and loading view
    vis.add_action("Save Camera View", save_view)
    vis.add_action("Load Camera View", load_view)

    # Create Random Geometry
    pc = np.random.randn(100, 3) * 0.5
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    vis.add_geometry("Random Point Cloud", pcd)
    vis.reset_camera_to_default()

    gui.Application.instance.run()


if __name__ == "__main__":
    main()