import numpy as np
import open3d as o3d
from open3d.visualization import gui
import random

NUM_LINES = 10
def random_point():
    return [5 * random.random(), 5 * random.random(), 5 * random.random()]

pts = [random_point() for _ in range(0, 2 * NUM_LINES)]
line_indices = [[2 * i, 2 * i + 1] for i in range(0, NUM_LINES)]
colors = [[0.0, 0.0, 0.0] for _ in range(0, NUM_LINES)]

lines = o3d.geometry.LineSet()
lines.points = o3d.utility.Vector3dVector(pts)
lines.lines = o3d.utility.Vector2iVector(line_indices)
# The default color of the lines is white, which will be invisible on the
# default white background. So we either need to set the color of the lines
# or the base_color of the material.
lines.colors = o3d.utility.Vector3dVector(colors)

gui.Application.instance.initialize()
win = o3d.visualization.O3DVisualizer("03DVisualizer",640, 480)
gui.Application.instance.add_window(win)
win.show_axes = True
#material = o3d.visualization.rendering.Material()
win.add_geometry("lines", lines) # flower is a point cloud
win.reset_camera_to_default()
import ipdb; ipdb.set_trace()
gui.Application.instance.run()