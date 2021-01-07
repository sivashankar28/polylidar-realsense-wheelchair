from os import path
from pathlib import Path
import logging
import warnings
from simplifyline import simplify_line_2d, MatrixDouble, simplify_radial_dist_2d, simplify_radial_dist_3d

logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from joblib import load
from scipy.spatial.transform import Rotation as R
from scipy.ndimage.filters import uniform_filter1d
from mpl_toolkits.mplot3d import Axes3D

DATA_DIR = Path('./data/scratch_test')

# Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def get_files():
    p = DATA_DIR.glob('**/*')
    files = sorted([x for x in p if x.is_file()])
    return files


def rotate_data_planar(points, normal):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rm, _ = R.align_vectors([[0, 0, 1]], [normal])  # Rotating matrix the polygon
    return rm.apply(points)


def plot_points(ax, X, c=None):
    if c is not None:
        ax.scatter(*np.split(X, X.shape[1], axis=1), c=c)
    else:
        ax.scatter(*np.split(X, X.shape[1], axis=1))

def setup_figure_3d():
    fig, ax_ = plt.subplots(nrows=1, ncols=1, subplot_kw=dict(projection='3d'))

    ax_.set_xlabel("X")
    ax_.set_ylabel("Y")
    ax_.set_zlabel("Z")

    return fig, ax_

def setup_figure_2d():
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    for ax_ in ax:
        ax_.set_xlabel("X")
        ax_.set_ylabel("Y")
        ax_.axis('equal')
    return fig, ax


def visualize_3d(first_points_rot, second_points_rot=None):
    fig, ax = setup_figure_3d()
    plot_points(ax, first_points_rot)
    if second_points_rot:
        plot_points(ax, second_points_rot)
    ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
    set_axes_equal(ax) # IMPORTANT - this is also required
    # ax.invert_zaxis()
    plt.show()


def filter_points(top_points, max_z=0.5, max_dist=0.05):
    top_points_simplified = np.array(simplify_radial_dist_3d(MatrixDouble(top_points), max_dist))
    # import ipdb; ipdb.set_trace()
    nearest_z = top_points_simplified[:, 2].min()
    far_z = nearest_z + max_z
    a1 = top_points_simplified[:, 2] < far_z

    # fig, ax = setup_figure_3d()
    # colors = np.ones((top_points.shape[0]))
    # colors[a1] = 0
    # plot_points(ax, top_points, c=colors)
    # ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
    # set_axes_equal(ax) # IMPORTANT - this is also required
    # plt.show()
    np_diff = np.diff(np.hstack(([False],a1==True,[False])))
    idx_pairs = np.where(np_diff)[0].reshape(-1,2)
    great_idx = np.diff(idx_pairs,axis=1).argmax()
    start_idx, end_idx = idx_pairs[great_idx, 0], idx_pairs[great_idx, 1]
    # print(a1)
    # print(np_diff)
    # print(idx_pairs)
    # print(start_idx, end_idx)
    # print(a1[start_idx], a1[end_idx])
    filtered_top_points = top_points_simplified[start_idx:end_idx, :]
    # need to roll it such that the jump starts on the first index
    # TODO you need to check if end_ind 

    # fig, ax = setup_figure_3d()
    # plot_points(ax, filtered_top_points)
    # ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
    # set_axes_equal(ax) # IMPORTANT - this is also required
    # plt.show(
    return filtered_top_points

def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

def get_rmse(poly1d_fn, points):
    predictions = poly1d_fn(points[:,0])
    targets = points[:, 1]
    return np.sqrt(np.mean((predictions-targets)**2))

def fit_line(points, idx):
    last_idx = min(idx[1] + 1, points.shape[0])
    points_ = points[idx[0]:last_idx, :]
    w = np.ones(points_.shape[0])
    w[0] = 0.25
    w[-1] = 0.75
    coef = np.polyfit(points_[:,0],points_[:,1],1, w=w)
    poly1d_fn = np.poly1d(coef)
    rmse = get_rmse(poly1d_fn, points_)
    return points_, poly1d_fn, rmse

def extract_lines(pc, window_size=5, dot_min=0.83):

    pc_shift = np.roll(pc, -1, axis=0)
    diff = pc_shift - pc
    diff_vec, length = normalized(diff)
    idx_max = np.argmax(length)
    assert idx_max != 0 or idx_max != length.shape[0] - 1, "LineString is not continuously connected"

    x = uniform_filter1d(diff[:, 0], size=window_size)
    y = uniform_filter1d(diff[:, 1], size=window_size)
    diff_smooth = np.column_stack((x,y))

    diff_smooth, length = normalized(diff)


    # print(diff)
    # print(diff_smooth)
    diff_smooth_shift = np.roll(diff_smooth, -1, axis=0)
    acos = np.einsum('ij, ij->i', diff_smooth, diff_smooth_shift)
    print(acos)


    mask = acos > dot_min
    np_diff = np.diff(np.hstack(([False],mask,[False])))
    idx_pairs = np.where(np_diff)[0].reshape(-1,2)
    print(idx_pairs)
    # Here I can do a more robust line fitting estimation, but for now
    # just fit all the lines
    fit_lines = [fit_line(pc, idx) for idx in idx_pairs if idx[1] - idx[0] > 1]
    return fit_lines

def plot_fit_lines(ax, fit_lines):
    next(ax._get_lines.prop_cycler)
    for fit_line in fit_lines:
        poly1d_fn = fit_line[1]
        points = fit_line[0]
        ax.plot(points[:, 0], poly1d_fn(points[:, 0]), '-')

def visualize_2d(top_points, top_normal):
    top_points_2d = rotate_data_planar(top_points, top_normal)[:, :2]
    fit_lines = extract_lines(top_points_2d)

    fig, ax = setup_figure_2d()
    plot_points(ax[0], top_points_2d)
    for i in range(top_points_2d.shape[0]):
        ax[0].annotate(str(i), (top_points_2d[i,0], top_points_2d[i, 1]))
    plot_fit_lines(ax[0], fit_lines)
    
    plt.show()


def choose_plane(data):
    first_centroid = data['first_plane']['point']
    second_centroid = data['second_plane']['point']
    first_normal = data['first_plane']['normal_ransac']

    proj_first = np.dot(first_centroid, first_normal)
    proj_second = np.dot(second_centroid, first_normal)

    if proj_second < proj_first:
        return data['second_plane']
    else:
        return data['first_plane']
    # print(first_normal)
    # print(proj_first, proj_second)

def process(data):
    """ Process the bottom and top planes dictionary """

    # first_points = data['first_plane']['all_points']
    # second_points = data['second_plane']['all_points']
    # first_normal = data['first_plane']['normal_ransac']
    # second_normal = data['first_plane']['normal_ransac']
    # first_points_rot = rotate_data_planar(first_points, first_normal)
    # second_points_rot = rotate_data_planar(second_points, second_normal)
    # visualize_3d(first_points, second_points)
    # if first_points_rot[0,2] > second_points_rot[0, 2]:
    #     print("second is top")
    #     top_points, top_normal, top_points_rot = second_points, second_normal, second_points_rot
        # logging.info("First Height %s, Second Height %s", first_points_rot[0,2], second_points_rot[0, 2])

    top_plane = choose_plane(data)
    top_points, top_normal = top_plane['all_points'], top_plane['normal_ransac']
    # visualize_3d(top_points)
    filtered_top_points = filter_points(top_points)
    visualize_2d(filtered_top_points, top_normal)



def main():
    files = get_files()
    for f in files:
        data = load(f)
        process(data)

if __name__ == "__main__":
    main()