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


def get_rmse(x_points, y_points, fn):
    predictions = fn(x_points)
    targets = y_points
    return np.sqrt(np.mean((predictions-targets)**2))


def fit_line(points, idx, max_slope=2.0):
    last_idx = min(idx[1] + 1, points.shape[0])
    points_ = points[idx[0]:last_idx, :]
    w = np.ones(points_.shape[0])
    w[0] = 0.25  # weight first and last a little  less
    w[-1] = 0.75

    dx = abs(points_[-1, 0] - points_[0, 0])
    dy = abs(points_[-1, 1] - points_[0, 1])
    flip_axis = False
    x_points = points_[:, 0]
    y_points = points_[:, 1]
    # print(dy, dx)
    if dy/dx > max_slope:
        flip_axis = True
        x_points = y_points
        y_points = points_[:, 0]

    coef = np.polyfit(x_points, y_points, 1, w=w)
    poly1d_fn = np.poly1d(coef)
    rmse = get_rmse(x_points, y_points, poly1d_fn)

    if flip_axis:
        run = 1 if x_points[-1] - x_points[0] > 0 else -1
        dir_vec = np.array([run * coef[0], run])
        logging.debug("Flip axis: %s, %s, %s", x_points[-1] - x_points[0], run, dir_vec)
    else:
        run = 1 if x_points[-1] - x_points[0] > 0 else -1
        dir_vec = np.array([run, coef[0] * run])
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    res = dict(points=points_, x_points=x_points, y_points=y_points, fn=poly1d_fn,
               rmse=rmse, dir_vec=dir_vec, idx=[idx[0], last_idx], flip_axis=flip_axis)
    logging.debug("Fit a new line: %s", res)
    return res


def extract_lines(pc, window_size=3, dot_min=0.88):

    pc_shift = np.roll(pc, -1, axis=0)
    diff = pc_shift - pc
    diff_vec, length = normalized(diff)
    idx_max = np.argmax(length)
    assert idx_max != 0 or idx_max != length.shape[0] - \
        1, "LineString is not continuously connected"

    x = uniform_filter1d(diff[:, 0], size=window_size)
    y = uniform_filter1d(diff[:, 1], size=window_size)
    diff_smooth = np.column_stack((x, y))

    diff_smooth, length = normalized(diff_smooth)
    diff_smooth_shift = np.roll(diff_smooth, -1, axis=0)
    acos = np.einsum('ij, ij->i', diff_smooth, diff_smooth_shift)
    # print(acos)
    mask = acos > dot_min
    np_diff = np.diff(np.hstack(([False], mask, [False])))
    idx_pairs = np.where(np_diff)[0].reshape(-1, 2)
    logging.debug("IDX Pairs %s", (idx_pairs))
    # print(diff_smooth)
    # I can do a more robust line fitting estimation, but for now
    # just fit all the lines
    fit_lines = [fit_line(pc, idx) for idx in idx_pairs if idx[1] - idx[0] > 1]
    # print(fit_lines)

    # print(diff)
    # print(diff_smooth)
    # print(acos)
    # print(idx_pairs)
    return fit_lines


def plot_fit_lines(ax, fit_lines):
    next(ax._get_lines.prop_cycler)
    for fit_line in fit_lines:
        poly1d_fn = fit_line['fn']
        if fit_line['flip_axis']:
            ax.plot(poly1d_fn(fit_line['x_points']), fit_line['x_points'], '-')
        else:  
            points = fit_line['points']
            ax.plot(points[:, 0], poly1d_fn(points[:, 0]), '-')


def orthogonal_distance(line_point, line_vec, points):
    #vec =  (a - p) - ((a-p) * n)*n
    # import ipdb; ipdb.set_trace()
    line_offset = points - line_point
    line_proj = line_offset @ line_vec
    line_vec_array = np.ones((line_proj.shape[0], 2)) * line_vec
    line_proj = np.multiply(
        line_vec_array, line_proj.reshape(line_proj.shape[0], 1))
    perf_offset = line_offset - line_proj
    _, lengths = normalized(perf_offset)
    median = np.median(lengths)
    # print(lengths, median)
    # get norm of vec
    return median


def check_merge_line(points, line, line_next, i, max_idx_dist=3, max_rmse=1.0, min_dot_prod=0.90, max_ortho_dist=0.05):
    idx_diff = line_next['idx'][0] - line['idx'][1]
    dot_prod = np.dot(line['dir_vec'], line_next['dir_vec'])
    logging.debug("attempting to merge line %s with %s, dot_prod: %s, idx_diff: %s", i, i+1, dot_prod, idx_diff)
    # print(line['dir_vec'], line_next['dir_vec'], dot_prod)
    if idx_diff < max_idx_dist and dot_prod > min_dot_prod:
        # its possible these two line segments should be refit to make a new line
        # combine points and refit
        logging.debug(
            "idx and dir vec passed! attempting to merge line %s with %s", i, i+1)
        new_idx = [line['idx'][0], line_next['idx'][1]]
        new_line = fit_line(points, new_idx)
        new_line_dot_prod = np.dot(line['dir_vec'], new_line['dir_vec'])
        new_line_ortho_dist1 = orthogonal_distance(
            line['points'].mean(axis=0), line['dir_vec'], line_next['points'])
        new_line_ortho_dist2 = orthogonal_distance(
            line_next['points'].mean(axis=0), line_next['dir_vec'], line['points'])
        new_line_ortho_dist = np.min(
            [new_line_ortho_dist1, new_line_ortho_dist2])
        # print("new line dot prod", new_line_dot_prod)
        if new_line_dot_prod > min_dot_prod and new_line_ortho_dist < max_ortho_dist:
            # new line still looks good! TODO RMSE check as well?
            logging.debug("Merging line %s with %s", i, i+1)
            return True, new_line
    return False, None


def merge_lines(points, lines, max_idx_dist=3, max_rmse=1.0, min_dot_prod=0.90, max_ortho_dist=0.05):
    final_lines = []
    i = 0
    last_line_added = False

    temporary_merged_line = None
    while i < (len(lines)):
        if temporary_merged_line is None:
            if i == (len(lines) - 1):
                final_lines.append(lines[i])
                break
            line = lines[i]
            line_next = lines[i + 1]

        else:
            line = temporary_merged_line
            line_next = lines[i]

        merged, new_line = check_merge_line(
            points, line, line_next, i, max_idx_dist, max_rmse, min_dot_prod, max_ortho_dist)
        if merged:
            i = i + 1 if temporary_merged_line else i + 2
            temporary_merged_line = new_line
        else:
            if temporary_merged_line is None:
                final_lines.append(line)
                i += 1
            else:
                final_lines.append(temporary_merged_line)
                temporary_merged_line = None

    if temporary_merged_line:
        final_lines.append(temporary_merged_line)

    return final_lines


def extract_lines_wrapper(top_points, top_normal, min_points_line=6):
    top_points_2d = rotate_data_planar(top_points, top_normal)[:, :2]
    all_fit_lines = extract_lines(top_points_2d)
    # print(all_fit_lines)
    best_fit_lines = merge_lines(top_points_2d, all_fit_lines)
    best_fit_lines = [
        fit_line for fit_line in best_fit_lines if fit_line['points'].shape[0] >= min_points_line]
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
    # visualize_3d(top_points)
    t1 = time.perf_counter()
    filtered_top_points = filter_points(top_points)  # < 500 us
    # extract_lines_wrapper(top_points, top_normal) # ~ 2-5ms
    t2 = time.perf_counter()
    # print(t1-t2)
    visualize_2d(filtered_top_points, top_normal)


def main():
    files = get_files()
    for idx, f in enumerate(files):
        if idx < 51:  # 40:
            continue
        logging.info("Processing %s", f)
        data = load(f)
        process(data)


if __name__ == "__main__":
    main()
