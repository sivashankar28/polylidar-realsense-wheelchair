from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import uniform_filter1d
from scipy.spatial.transform import Rotation as R
from joblib import load
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from surfacedetector.utility.helper_general import set_axes_equal, rotate_data_planar, plot_points, setup_figure_2d, setup_figure_3d
from simplifyline import simplify_line_2d, MatrixDouble, simplify_radial_dist_2d, simplify_radial_dist_3d
from os import path
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)


DATA_DIR = Path('./data/scratch_test')


def get_files():
    p = DATA_DIR.glob('*curbmodel2*')
    files = sorted([x for x in p if x.is_file()])
    return files


def visualize_3d(first_points_rot, second_points_rot=None):
    fig, ax = setup_figure_3d()
    plot_points(ax, first_points_rot)
    if second_points_rot:
        plot_points(ax, second_points_rot)
    ax.set_box_aspect([1, 1, 1])  # IMPORTANT - this is the new, key line
    set_axes_equal(ax)  # IMPORTANT - this is also required
    # ax.invert_zaxis()
    plt.show()


def filter_points(top_points, max_z=0.5, max_dist=0.05):
    top_points_simplified = np.array(
        simplify_radial_dist_3d(MatrixDouble(top_points), max_dist))
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
    np_diff = np.diff(np.hstack(([False], a1 == True, [False])))
    idx_pairs = np.where(np_diff)[0].reshape(-1, 2)
    great_idx = np.diff(idx_pairs, axis=1).argmax()
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
    predictions = poly1d_fn(points[:, 0])
    targets = points[:, 1]
    return np.sqrt(np.mean((predictions-targets)**2))


def fit_line(points, idx):
    last_idx = min(idx[1] + 1, points.shape[0])
    points_ = points[idx[0]:last_idx, :]
    w = np.ones(points_.shape[0])
    w[0] = 0.25  # weight first and last a little  less
    w[-1] = 0.75
    coef = np.polyfit(points_[:, 0], points_[:, 1], 1, w=w)
    poly1d_fn = np.poly1d(coef)
    rmse = get_rmse(poly1d_fn, points_)
    return dict(points=points_, fn=poly1d_fn, rmse=rmse)


def extract_lines(pc, window_size=5, dot_min=0.83):

    pc_shift = np.roll(pc, -1, axis=0)
    diff = pc_shift - pc
    diff_vec, length = normalized(diff)
    idx_max = np.argmax(length)
    assert idx_max != 0 or idx_max != length.shape[0] - \
        1, "LineString is not continuously connected"

    x = uniform_filter1d(diff[:, 0], size=window_size)
    y = uniform_filter1d(diff[:, 1], size=window_size)
    diff_smooth = np.column_stack((x, y))

    diff_smooth, length = normalized(diff)
    diff_smooth_shift = np.roll(diff_smooth, -1, axis=0)
    acos = np.einsum('ij, ij->i', diff_smooth, diff_smooth_shift)

    mask = acos > dot_min
    np_diff = np.diff(np.hstack(([False], mask, [False])))
    idx_pairs = np.where(np_diff)[0].reshape(-1, 2)
    # I can do a more robust line fitting estimation, but for now
    # just fit all the lines
    fit_lines = [fit_line(pc, idx) for idx in idx_pairs if idx[1] - idx[0] > 1]

    # print(diff)
    # print(diff_smooth)
    # print(acos)
    # print(idx_pairs)
    return fit_lines


def plot_fit_lines(ax, fit_lines):
    next(ax._get_lines.prop_cycler)
    for fit_line in fit_lines:
        poly1d_fn = fit_line['fn']
        points = fit_line['points']
        ax.plot(points[:, 0], poly1d_fn(points[:, 0]), '-')


def extract_lines_wrapper(top_points, top_normal, min_points_line=6):
    top_points_2d = rotate_data_planar(top_points, top_normal)[:, :2]
    all_fit_lines = extract_lines(top_points_2d)
    best_fit_lines = [
        fit_line for fit_line in all_fit_lines if fit_line['points'].shape[0] >= min_points_line]
    return top_points_2d, all_fit_lines, best_fit_lines


def visualize_2d(top_points, top_normal, min_points_line=6):
    top_points_2d, all_fit_lines, best_fit_lines = extract_lines_wrapper(
        top_points, top_normal, min_points_line)
    fig, ax = setup_figure_2d()
    plot_points(ax[0], top_points_2d)
    for i in range(top_points_2d.shape[0]):
        ax[0].annotate(str(i), (top_points_2d[i, 0], top_points_2d[i, 1]))
    plot_fit_lines(ax[0], all_fit_lines)
    plot_fit_lines(ax[1], best_fit_lines)
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


def process(data):
    """ Process the bottom and top planes dictionary """

    top_plane = choose_plane(data)
    top_points, top_normal = top_plane['all_points'], top_plane['normal_ransac']
    visualize_3d(top_points)
    # t1 = time.perf_counter()
    filtered_top_points = filter_points(top_points)
    # t2 = time.perf_counter()
    # print(t1-t2)
    visualize_2d(filtered_top_points, top_normal)


def main():
    files = get_files()
    for f in files:
        data = load(f)
        process(data)


if __name__ == "__main__":
    main()
