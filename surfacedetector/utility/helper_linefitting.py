"""Multi Line Fitting for Continuous Line Strings
This module contains functions which takes a noisy 3D line string, that may have multiple straight paths, 
and attempt to return geomerically lines to approximate those straight paths. The expectionion is that the 3D
Line string is actually planar in some dimension (like a flat surface)
For example, image a noisy square, 4 (3D) lines should be returned

Process:
1. Filter Points -  Uses a C++ module I created which first performs a form of line simplification
                    Next, remove points which are two far away from a certian axis. Helps focus on nearby points to observer.
                    The returned line is still continuous
2. Rotate -         Rotate the 3D line such that the XY plane becomes aligned with the 3D plane
3. Extract Lines -  Calculate the vectors for each line segment
                    Perform smoothing on the these vectors using a rolling window
                    Create continuous segments of points whose direction vectors are closely aligned
                    Fit lines to these segments (carefully handle vertical lines)
                    Filter out any lines which are too small
4. Merge Lines  -   Merge lines which are close to each other and which make a better fit. 
                    Metrics such as the dot angle, and their orhtogonal distance form eachother are used
                    to determine merging
5. Filter Lines -   Optional. Only return the pair of lines which are most orthogonal to each other.
"""
import time
import math
import sys
import logging
from itertools import combinations

import numpy as np
from numpy.lib.polynomial import poly
from scipy import cluster
from scipy.ndimage.filters import uniform_filter1d
from scipy.spatial.transform import Rotation as R
from simplifyline import MatrixDouble, simplify_radial_dist_3d
from surfacedetector.utility.helper_general import rotate_data_planar, normalized, bounding_elipse
import matplotlib.pyplot as plt
import matplotlib
from pprint import pprint
from scipy.cluster.hierarchy import linkage, fcluster

from surfacedetector.utility.helper_general import setup_figure_2d, plot_fit_lines, plot_points
from surfacedetector.utility.AngleAnnotation import AngleAnnotation

from matplotlib.transforms import Bbox
from matplotlib import colors as mcolors
TABLEAU_COLORS = {k: mcolors.to_rgba(v) for (k,v) in mcolors.TABLEAU_COLORS.items()}

def full_extent(ax, fig, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    fig.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    items += [ax.get_xaxis().get_label(), ax.get_yaxis().get_label()]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)


def pad_extent(ax, main_extent, pad=0.0):
    items = [ax, ax.title]
    items += [ax.get_xaxis().get_label(), ax.get_yaxis().get_label()]
    item_extents = [item.get_window_extent() for item in items]
    # item_extents.append(main_extent)
    bbox = Bbox.union(item_extents)
    return bbox.expanded(1.0 + pad, 1.0 + pad)

def choose_plane(first_plane, second_plane):
    first_centroid = first_plane['point']
    second_centroid = second_plane['point']
    first_normal = first_plane['normal_ransac']

    proj_first = np.dot(first_centroid, first_normal)
    proj_second = np.dot(second_centroid, first_normal)

    if proj_second < proj_first:
        return second_plane, first_plane
    else:
        return first_plane, second_plane


def filter_points(top_points, max_z=0.5, max_dist=0.05, min_z=0.6):
    """Filters and simplifies 3D points belonging to a continuos line string.
        Also filter points that are are less than min_z distance away

    Args:
        top_points (ndarray): 3D Line String
        max_z (float, optional): Will filter out all points who z value is greater than min(points,z) + max_z. Defaults to 0.5.
        max_dist (float, optional): Remove points closer than max_dist from eachother, cheap and fast simplification. Defaults to 0.05.

    Returns:
        [ndarray]: Filtered 3D line string
    """
    top_points_simplified = np.array(
        simplify_radial_dist_3d(MatrixDouble(top_points), max_dist))
    nearest_z = top_points_simplified[:, 2].min()
    far_z = nearest_z + max_z
    a1 = (top_points_simplified[:, 2] < far_z) & (top_points_simplified[:, 2] > min_z)

    np_diff = np.diff(np.hstack(([False], a1 == True, [False])))
    idx_pairs = np.where(np_diff)[0].reshape(-1, 2)
    great_idx = np.diff(idx_pairs, axis=1).argmax()
    start_idx, end_idx = idx_pairs[great_idx, 0], idx_pairs[great_idx, 1]

    filtered_top_points = top_points_simplified[start_idx:end_idx, :]
    # need to roll it such that the jump starts on the first index
    return filtered_top_points

def filter_points_from_wheel_chair(top_points, max_dist=0.05, max_planar_distance=0.9,
                                    wheel_chair_position=[0,0,0]):
    """Filters and simplifies 3D points belonging to a continuos line string.
        Also filter points that are are less than min_z distance away

    Args:
        top_points (ndarray): 3D Line String
        max_z (float, optional): Will filter out all points who z value is greater than min(points,z) + max_z. Defaults to 0.5.
        max_dist (float, optional): Remove points closer than max_dist from eachother, cheap and fast simplification. Defaults to 0.05.

    Returns:
        [ndarray]: Filtered 3D line string
    """
    # TODO FIX simplify_radial_dist_3d (memory corruption requiring a copy)
    top_points_simplified = np.copy(np.array(
        simplify_radial_dist_3d(MatrixDouble(top_points), max_dist)))

    # if top_points_simplified.shape[0] < 3 or np.count_nonzero(top_points_simplified) < 3:
    #     print(top_points)
    #     print(top_points_simplified)
    #     raise ValueError('Not enough points after simplification')

    dist_from_wc = np.linalg.norm(top_points_simplified[1:,:] - wheel_chair_position, axis=1) #TODO fix weird bug here in simplification
    nearest_dist_from_wc = dist_from_wc.min()
    far_dist = nearest_dist_from_wc + max_planar_distance
    a1 = (dist_from_wc < far_dist) & (dist_from_wc >= nearest_dist_from_wc)
    # import ipdb; ipdb.set_trace()

    np_diff = np.diff(np.hstack(([False], a1 == True, [False])))
    idx_pairs = np.where(np_diff)[0].reshape(-1, 2)
    great_idx = np.diff(idx_pairs, axis=1).argmax()
    start_idx, end_idx = idx_pairs[great_idx, 0], idx_pairs[great_idx, 1]

    filtered_top_points = top_points_simplified[start_idx:end_idx, :]

    if filtered_top_points.shape[0] < 3 or np.count_nonzero(filtered_top_points) < 3:
        print(top_points)
        print(top_points_simplified)
        raise ValueError('Not enough points after simplification and segmenting')
    # need to roll it such that the jump starts on the first index
    return filtered_top_points


def get_rmse(x_points, y_points, fn):
    """Get Root Mean Squared Error

    Args:
        x_points (ndarray): X Data
        y_points (ndarray): Y Data
        fn (function): Some function that takes X and predicts Y

    Returns:
        float: RMSE
    """
    predictions = fn(x_points)
    targets = y_points
    return np.sqrt(np.mean((predictions-targets)**2))


def fit_line(points, idx, max_slope=2.0, **kwargs):
    """Will fit a line to the points specified by index

    Args:
        points (ndarray): Point Set (2D)
        idx (List): Start and End idx in points for line
        max_slope (float, optional): If slope is greater than max_slope, flip x and y axis. Can't fit vertical lines. Defaults to 2.0.

    Returns:
        dict: dict(points, x_points, y_points, fn, rmse, dir_vec, idx, flip_axis)
    """

    t1 = time.perf_counter()
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
    if dy/dx > max_slope:
        flip_axis = True
        x_points = y_points
        y_points = points_[:, 0]
    t2 = time.perf_counter()
    # this is taking the most time
    coef = np.polyfit(x_points, y_points, 1, w=w)
    poly1d_fn = np.poly1d(coef)
    t3 = time.perf_counter()
    # this too
    rmse = get_rmse(x_points, y_points, poly1d_fn)
    t4 = time.perf_counter()

    ms1 = (t2-t1) * 1000
    ms2 = (t3-t2) * 1000
    ms3 = (t4-t3) * 1000
    logging.debug(
        "Fit Line - Get Points: %.2f, Regress Line: %.2f, RMSE: %.2f", ms1, ms2, ms3)
    if flip_axis:
        run = 1 if x_points[-1] - x_points[0] > 0 else -1
        dir_vec = np.array([run * coef[0], run])
        logging.debug("Flip axis: %s, %s, %s",
                      x_points[-1] - x_points[0], run, dir_vec)
    else:
        run = 1 if x_points[-1] - x_points[0] > 0 else -1
        dir_vec = np.array([run, coef[0] * run])
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    res = dict(points=points_, x_points=x_points, y_points=y_points, fn=poly1d_fn,
               rmse=rmse, dir_vec=dir_vec, idx=[idx[0], last_idx], flip_axis=flip_axis, points_2d_orig=points)
    logging.debug("Fit a new line: %s", res)
    return res


def extract_lines(pc, window_size=3, dot_min=0.95, **kwargs):
    """Extract a first approximation of all lines in 2D line string

    Args:
        pc (ndarray): 2D Line String
        window_size (int, optional): Smoothing window for Line. Defaults to 3.
        dot_min (float, optional): Minimum dot prodcut for joining lines. Defaults to 0.88.

    Returns:
        List: List of lines(dict)
    """

    t1 = time.perf_counter()
    pc_shift = np.roll(pc, -1, axis=0)
    diff = pc_shift - pc
    diff_vec, length = normalized(diff)
    idx_max = np.argmax(length)
    assert idx_max != 0 or idx_max != length.shape[0] - \
        1, "LineString is not continuously connected"
    t2 = time.perf_counter()
    x = uniform_filter1d(diff[:, 0], size=window_size)
    y = uniform_filter1d(diff[:, 1], size=window_size)
    diff_smooth = np.column_stack((x, y))
    t3 = time.perf_counter()

    diff_smooth, length = normalized(diff_smooth)
    diff_smooth_shift = np.roll(diff_smooth, -1, axis=0)
    acos = np.einsum('ij, ij->i', diff_smooth, diff_smooth_shift)
    t4 = time.perf_counter()

    mask = acos > dot_min
    np_diff = np.diff(np.hstack(([False], mask, [False])))
    idx_pairs = np.where(np_diff)[0].reshape(-1, 2)

    t5 = time.perf_counter()
    logging.debug("IDX Pairs %s", (idx_pairs))
    fit_lines = [fit_line(pc, idx) for idx in idx_pairs if idx[1] - idx[0] > 1]
    t6 = time.perf_counter()

    # ms1 = (t2-t1) * 1000
    # ms2 = (t3-t2) * 1000
    # ms3 = (t4-t3) * 1000
    # ms4 = (t5-t4) * 1000
    # ms5 = (t6-t5) * 1000
    # print(ms1, ms2, ms3, ms4, ms5)
    return fit_lines


def orthogonal_distance(line_point, line_vec, points, return_median=True):
    """Computes Orthogonal distance between a line and points

    Args:
        line_point (ndarray (2,)): Point
        line_vec (ndarray (2,)): Direction Vector of Line
        points (ndarray): Points

    Returns:
        [float]: Median Distance of points to line
    """
    #vec =  (a - p) - ((a-p) * n)*n
    line_offset = points - line_point
    line_proj = line_offset @ line_vec
    line_vec_array = np.ones((line_proj.shape[0], 2)) * line_vec
    line_proj = np.multiply(
        line_vec_array, line_proj.reshape(line_proj.shape[0], 1))
    perf_offset = line_offset - line_proj
    dir_vec, lengths = normalized(perf_offset)
    if return_median:
        return np.median(lengths)
    else:
        return dir_vec, lengths

    


def check_merge_line(points, line, line_next, i, max_idx_dist=5, max_rmse=1.0, min_dot_prod=0.93, max_ortho_dist=0.05, **kwargs):
    idx_diff = line_next['idx'][0] - line['idx'][1]
    dot_prod = np.dot(line['dir_vec'], line_next['dir_vec'])
    logging.debug("attempting to merge line %s with %s, dot_prod: %s, idx_diff: %s",
                  i, i+1, dot_prod, idx_diff)
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
        if new_line_dot_prod > min_dot_prod and new_line_ortho_dist < max_ortho_dist:
            # new line still looks good! TODO RMSE check as well?
            logging.debug("Merging line %s with %s", i, i+1)
            return True, new_line
    return False, None


def merge_lines(points, lines, max_idx_dist=5, max_rmse=1.0, min_dot_prod=0.93, max_ortho_dist=0.05, **kwargs):
    """Merges lines that may be close to each other

    Args:
        points (ndarray): Original 2D Point Line String
        lines (List[dict]): List of lines
        max_idx_dist (int, optional): Max distance in indices for joining two nearby lines. Defaults to 3.
        max_rmse (float, optional): Maximum RMSE of a new line fit. Defaults to 1.0.
        min_dot_prod (float, optional): Minimum dot prodcut between two lines to being joines. Defaults to 0.90.
        max_ortho_dist (float, optional): Max orthognal distance for two lines to being joined. Defaults to 0.05.

    Returns:
        List[dict]: List of lines
    """
    final_lines = []
    i = 0
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


def sort_lines(best_fit_lines, w1=0.75, w2=0.25, wheel_chair_direction_vec_sensor_frame=[0,0,1], too_close=0.5):
    max_dist = max(best_fit_lines, key=lambda a: a['dist_to_line'])['dist_to_line'] + 0.5
    for line in best_fit_lines:
        dist_to_line = line['dist_to_line']
        dist_to_line_norm = dist_to_line / max_dist
        ang_dist_norm = np.abs(np.dot(wheel_chair_direction_vec_sensor_frame, line['hplane_normal']))
        sort_metric = (1 - dist_to_line_norm) * w1 +  ang_dist_norm * w2
        # print(f"Dist to line: {dist_to_line:.2f}, dist to line norm: {dist_to_line_norm:.2f}, ang_dist_norm: {ang_dist_norm:.2f},  sort_metric: {sort_metric:.2f}")
        if dist_to_line < too_close:
            sort_metric -= 1
        line['sort_metric'] = sort_metric
    # pprint(best_fit_lines)
    return sorted(best_fit_lines, key=lambda i: i['sort_metric'], reverse=True)


def filter_lines(best_fit_lines, max_dot=0.2, w1=0.75, w2=0.25, return_only_one_line=True, 
                 wheel_chair_direction_vec_sensor_frame=[0,0,1], **kwargs):
    """Filter lines to a max of 2, they must be orthogonal
    If multiple pairs can be found, find the one that maximizes:
    metric = w1 * (1- dot_prod) * w2 * idx_length
    Basically the pair of line strings that are the longest and most orthogonal

    Args:
        best_fit_lines (List): List of Lines
        max_dot (float, optional): Max dot product if more than two lines. Defaults to 0.2.
        w1 (float, optional): Weight of dot product. Defaults to 0.75.
        w2 (float, optional): Weight of Line Length. Defaults to 0.25

    Returns:
        [type]: [description]
    """
    best_pair = []
    if return_only_one_line and best_fit_lines:
        # only return one line, choose the closest line
        # TODO RMSE and Length metric as well
        # for line in best_fit_lines:
        #     line['distance'] = np.linalg.norm(line['hplane_point'])
        sorted_lines = sort_lines(best_fit_lines, w1=w1, w2=w2, wheel_chair_direction_vec_sensor_frame=wheel_chair_direction_vec_sensor_frame )
        return sorted_lines[0:1]
    elif len(best_fit_lines) <= 1:
        best_pair = best_fit_lines
    else:
        best_metric = 0.0
        max_line_length = float(
            max([line['points'].shape[0] for line in best_fit_lines]))
        all_line_set = combinations(best_fit_lines, 2)
        for line1, line2 in all_line_set:
            dot_prod = abs(np.dot(line1['dir_vec'], line2['dir_vec']))
            length = ((line1['points'].shape[0] +
                       line2['points'].shape[0]) / 2.0) / max_line_length
            metric = w1 * (1 - dot_prod) + w2 * length
            if metric > best_metric and dot_prod < np.abs(max_dot):
                best_pair = [line1, line2]
                best_metric = metric

    sorted_lines = sort_lines(best_fit_lines, w1=w1, w2=w2, wheel_chair_direction_vec_sensor_frame=wheel_chair_direction_vec_sensor_frame )
    return sorted_lines

    # for line in best_fit_lines:
    #     line['distance'] = np.linalg.norm(line['hplane_point'])
    # return sorted(best_pair, key=lambda i: i['distance'])


def make_square(cent, ax1, ax2, normal, w=0.3, h=0.25):
    p1 = cent - w * ax2
    p2 = cent + w * ax2
    p3 = cent + h * ax1 + w * ax2
    p4 = cent + h * ax1 - w * ax2
    points = np.array([p1, p2, p3, p4])
    # projected_points = project_points_geometric_plane(points, normal, cent)
    projected_points = points
    return projected_points


def project_points_geometric_plane(points, normal, point_on_plane):
    diff = points - point_on_plane
    dist = np.dot(diff, normal)
    scaled_vector = normal*dist[:, np.newaxis]
    projected_points = points - scaled_vector

    return projected_points


def recover_3d_lines(best_fit_lines, top_normal, height, 
                     wheel_chair_origin_sensor_frame=[-0.34, 0, 0], curb_height=0.5):
    """This will recover 3D information of the lines

    Args:
        best_fit_lines (List[dict]): List of lines
        top_normal (ndarray): The normal of the plane which all lines are coplanar with
        height (float): Height of the plane in z direction

    Returns:
        List[dict]: Same list, but augmented with more data
    """
    for line in best_fit_lines:
        pts = line['points']
        points_2d_orig = line['points_2d_orig']
        line['points_3d'] = np.append(
            pts, np.ones((pts.shape[0], 1)) * height, axis=1)
        line['points_3d'] = rotate_data_planar(
            line['points_3d'], top_normal, True)
        line['points_3d_orig'] = np.append(
            points_2d_orig, np.ones((points_2d_orig.shape[0], 1)) * height, axis=1)
        line['points_3d_orig'] = rotate_data_planar(
            line['points_3d_orig'], top_normal, True)
        line['dir_vec_3d'] = np.array(
            [[line['dir_vec'][0], line['dir_vec'][1], 0]])
        line['dir_vec_3d'] = rotate_data_planar(
            line['dir_vec_3d'], top_normal, True).flatten()
        line['dir_vec_3d'] = line['dir_vec_3d'] / \
            np.linalg.norm(line['dir_vec_3d'])
        line['plane_normal'] = top_normal
        multiply = 1.0 if np.array_equal(top_normal, [0.0, 0.0, 1.0]) else -1.0
        line['hplane_normal'] = np.cross(
            line['dir_vec_3d'], line['plane_normal']) * multiply
        line['hplane_normal'] = line['hplane_normal'] / \
            np.linalg.norm(line['hplane_normal'])
        line['hplane_point'] = line['points_3d'].mean(axis=0)
        line['square_points'] = make_square(
            line['hplane_point'], line['plane_normal'] * -multiply, line['dir_vec_3d'], line['hplane_normal'], h=curb_height)
        line['dist_to_line'] = np.linalg.norm(line['hplane_point'] - wheel_chair_origin_sensor_frame)

    return best_fit_lines


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

def extract_lines_wrapper(top_points, top_normal, min_points_line=12, **kwargs):
    t1 = time.perf_counter()

    top_points_3d = rotate_data_planar(top_points, top_normal)
    top_points_2d = top_points_3d[:, :2]

    height = np.mean(top_points_3d[:, 2])
    t2 = time.perf_counter()
    all_fit_lines = extract_lines(top_points_2d, **kwargs)
    t3 = time.perf_counter()
    best_fit_lines = merge_lines(top_points_2d, all_fit_lines, **kwargs)
    t4 = time.perf_counter()
    ms1 = (t2-t1) * 1000
    ms2 = (t3-t2) * 1000
    ms3 = (t4-t3) * 1000
    logging.debug(
        "Extract lines Wrapper - Rotate: %.2f, Extract Lines: %.2f, Merge Lines: %.2f", ms1, ms2, ms3)
    best_fit_lines = [
        fit_line for fit_line in best_fit_lines if fit_line['points'].shape[0] >= min_points_line]
    best_fit_lines = recover_3d_lines(best_fit_lines, top_normal, height)
    best_fit_lines = filter_lines(best_fit_lines, **kwargs)

    return top_points_2d, height, all_fit_lines, best_fit_lines

def get_point_clusters(points, clusters):
    """Cluster points (R3) together given a cluster grouping"""
    point_clusters = []
    cluster_groups = np.unique(clusters)
    for cluster in cluster_groups:
        temp_mask = clusters == cluster
        point_clusters.append(points[temp_mask, :])
        # point_clusters.append((points[temp_mask, :], point_weights[temp_mask]))
    return point_clusters


def average_clusters(points, clusters, min_num_models=3):
    """Average any clusters together by weights, remove any that don't meet a minimum requirements"""
    cluster_points = get_point_clusters(points, clusters)
    clusters_averaged = []
    cluster_idx = []
    for clust_idx, points in enumerate(cluster_points):
        if points.shape[0] >= min_num_models:
            avg_point = np.average(points, axis=0)
            clusters_averaged.append(avg_point)
            cluster_idx.append(clust_idx + 1)

    clusters_filtered = np.array(clusters_averaged)
    return clusters_filtered, cluster_idx


def cluster_lines(points, cluster_kwargs=dict(t=0.10, criterion='distance')):
    Z = linkage(points, 'single')
    clusters = fcluster(Z, **cluster_kwargs)
    return clusters


def create_line_model(line_point, line_vec, points, cluster_idx, max_slope=2.0):
    flip_axis = False
    x_points = points[:, 0]
    y_points = points[:, 1]
    m = line_vec[1] / line_vec[0]
    b = line_point[1] - m * line_point[0]
    if m > max_slope:
        flip_axis = True
        x_points = y_points
        y_points = points[:, 0]
        m = line_vec[0] / line_vec[0]
        b = line_point[0] - m * line_point[1]
    coef = [m, b]

    poly1d_fn = np.poly1d(coef)
    res = dict(points=points, x_points=x_points, y_points=y_points, fn=poly1d_fn, cluster_idx=cluster_idx,
            dir_vec=line_vec, line_point=line_point, flip_axis=flip_axis, points_2d_orig=points, rmse=0.0)

    return res

def evaluate_and_filter_models(lines, max_ortho_offset=0.05, min_inlier_ratio=0.15):
    filtered_line_models = []
    for line in lines:
        line_point = line['line_point']
        line_vec = line['dir_vec']
        all_points = line['points']
        # save for reference later....
        line['dir_vec_old'] = line_vec
        line['line_point_old'] = line_point

        total_points = all_points.shape[0]
        ortho_dir, ortho_dist = orthogonal_distance(line_point, line_vec, all_points, return_median=False)
        mask = ortho_dist < max_ortho_offset
        points_ = all_points[mask, :]
        inlier_abs_dev = ortho_dist[mask]

        line['ortho_dir'] = ortho_dir
        line['ortho_dist'] = ortho_dist
        line['ortho_dir_dist'] = ortho_dir * ortho_dist[:, np.newaxis]
        line['inlier_mask'] = mask

        # print(f"For LP: {line_point}; line_vec: {line_vec}")
        # pprint(ortho_dist)
        num_inliers = points_.shape[0]
        inlier_ratio = float(num_inliers)/float(total_points)
        if inlier_ratio < min_inlier_ratio:
            continue
        
        x_points = points_[:, 0]
        y_points = points_[:, 1]

        w = np.ones(x_points.shape[0])
        w[0] = 0.25  # weight first and last a little  less
        w[-1] = 0.25

        if line['flip_axis']:
            y_points = x_points
            x_points = points_[:, 1]
        coef = np.polyfit(x_points, y_points, 1, w=w)
        poly1d_fn = np.poly1d(coef)
        
        line['fn_old'] = line['fn']
        line['fn'] = poly1d_fn
        line['x_points'] = x_points
        line['y_points'] = y_points
        line['points'] = points_
        line['all_points'] = all_points
        line['rmse'] = np.sqrt(np.sum((inlier_abs_dev **2)) / inlier_abs_dev.shape[0])

        # TODO dir_vec should be updated
        y_points = poly1d_fn(x_points)
        p1 = np.array([x_points[0], y_points[0]])
        p2 = np.array([x_points[-1], y_points[-1]])

        dir_vec = (p2 - p1)
        dir_vec = dir_vec / np.linalg.norm(dir_vec)

        # print("Original ", line['dir_vec'])
        if line['flip_axis']:
            dir_vec = np.arrray([dir_vec[1], dir_vec[0]])
        if dir_vec[0] < 0:
            dir_vec *= -1.0
        # print("Refit ", dir_vec)
        line['dir_vec'] = dir_vec

        filtered_line_models.append(line)
    return filtered_line_models


def set_up_axes(ax):
    ax.set_xlabel(r"X (m)")
    ax.set_ylabel(r"Y (m)")
    ax.axis("equal")




def extract_lines_parameterized(pc, idx_skip=2, window_size=4, 
                                cluster_kwargs=dict(t=0.10, criterion='distance'),
                                min_num_models=3, max_ortho_offset=0.05, min_inlier_ratio=0.20, 
                                debug=False, max_origin_offset=2.5, min_origin_offset=0.5, **kwargs):
    """Extract possible lines within line segment point cloud
    0. Dowsample point cloud segment by skipping over points with by index (idx_skip)
    1. Create vectors for each line segment. These vectors are simple proposed line models!
    2. Smooth these vectors using a uniform filter (window_size)
    3. Represent these line models (smoothed vectors) in a parameter space (angle and origin offset) as 2D points.
    4. Cluster these points (which are line models!) using agglomerative clustering to find best "average" line models. 
    5. Filter the proposed "average" line models by inlier ratio from ALL points. Refit the line models with inliers to create a best fit line.
    """
    np.set_printoptions(precision=4, suppress=True)
    t1 = time.perf_counter()
    pc_skip = pc[::idx_skip, :] # skip point to reduce noise
    # 1. Create vectors for each line segment
    pc_shift = np.roll(pc_skip, -1, axis=0) # shift point cloud idx to right
    diff = pc_shift - pc_skip # create vector diff between consecutive points
    diff_vec, length = normalized(diff) 
    idx_max = np.argmax(length)
    assert idx_max != 0 or idx_max != length.shape[0] - \
        1, "LineString is not continuously connected"

    # 2. Smooth vector estimate of line
    skip_window_edge = int(np.ceil(window_size / 2.0)) # must remove points at edge of smoothed window, will give erroneous result if kept
    t2 = time.perf_counter()
    x = uniform_filter1d(diff[:, 0], size=window_size)
    y = uniform_filter1d(diff[:, 1], size=window_size)
    diff_smooth = np.column_stack((x, y))
    diff_smooth_filt = diff_smooth[skip_window_edge-1:-skip_window_edge]
    t3 = time.perf_counter()

    # 3. Represent these line models (smoothed vectors) in a parameter space, angle and origin offset
    #    Neat Trick - Convert angle to postion on unit sphere (normalized) and then scale with origin offset
    #    Now we have a single 2D point in Euclidean space that represents the line angle and origin offset. Points close together mean similar lines!
    # MidPoint of estimated line
    mid_point = ((pc_shift + pc_skip)/2.0)[skip_window_edge-1:-skip_window_edge, :]
    # Unit vector of angle estimate of line direction
    line_vec_norm, _ = normalized(diff_smooth_filt)
    line_vec_norm_orig = np.copy(line_vec_norm)
    mask = line_vec_norm[:, 0] < 0
    line_vec_norm[mask, :] = -1 * line_vec_norm[mask, :] # change directionality of line_vec_norm if angle is negative value (always point +x)
    # convert to original unit vector of line angle. Basically this unit vector of a line starting from origin that orthogonally intersects with the line estimate
    # this is done as a 90 degree rotation of the line vector
    rot = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2.0)], [np.sin(np.pi/2.0), np.cos(np.pi/2.0)]])
    ang_vec_norm = np.matmul(line_vec_norm, rot.transpose())

    # Now we need to get orthogonal offset of this line from origin
    origin_offset = np.einsum('ij,ij->i', line_vec_norm, mid_point)[:, np.newaxis] * line_vec_norm - mid_point
    origin_offset = np.linalg.norm(origin_offset, axis=1)
    t4 = time.perf_counter()
    # Representing a line as a point in 2D space. All Euclidean Space, but same representation as angle offset parameter space
    # Scale origin offset such that angles the average offset is 1 meters.
    origin_offset_scaled = np.clip(origin_offset, min_origin_offset, max_origin_offset)
    condensed_param_set = ang_vec_norm * origin_offset_scaled[:, np.newaxis] # used for clustering
    condensed_param_set_true = ang_vec_norm * origin_offset[:, np.newaxis] # true parametrization of line angle and origin offset

    # 4. Cluster these points (which are line models!) using agglomerative clustering to find best "average" line models. 
    #    Only allow clusters with > min_num_models. Average models in a cluster to get final line model for that specific cluster.
    # cluster similar lines by point distance
    try:
        clusters = cluster_lines(condensed_param_set, cluster_kwargs=cluster_kwargs)
        cluster_average, cluster_idx = average_clusters(condensed_param_set_true, clusters, min_num_models=min_num_models)
        t5 = time.perf_counter()
    except Exception as e:
        logging.exception("Something went wrong during clustering")
        pprint(pc)
        pprint(condensed_param_set)
        raise

    # Decompose the angle and the origin offset
    ang_vec_norm_cluster, _ = normalized(cluster_average)
    # recover the line vector, reverse 90 degree rotation
    line_vec_norm_cluster = np.matmul(ang_vec_norm_cluster, rot)

    # 5. Filter the proposed "average" line models by inlier ratio from ALL points. Refit the line models with inliers to create a best fit line.
    line_models = [create_line_model(cluster_average[i,:], line_vec_norm_cluster[i, :], pc, cluster_idx[i]) for i in range(cluster_average.shape[0])]
    line_models_filtered = evaluate_and_filter_models(line_models, max_ortho_offset=max_ortho_offset, min_inlier_ratio=min_inlier_ratio)

    if debug:
        # Angle Offset Parameter Space, not good for clustering, just showing for comparison in vis.
        # This if just for plotting, same info but in radians instead of (x,y) pont of circle
        deg_ang = np.degrees(np.arctan2(ang_vec_norm[:, 1], ang_vec_norm[:, 0]))
        deg_ang_cluster = np.degrees(np.arctan2(line_vec_norm_cluster[:, 1], line_vec_norm_cluster[:, 0]))
        # deg_ang = deg_ang + 90
        # mask = deg_ang < 0
        # deg_ang[mask] += 360.0
        # deg_ang = np.unwrap(deg_ang)
        parameter_set = np.column_stack((deg_ang, origin_offset))

        # Idea, map each proposed line as a new color, colored by index
        cmap_viridis = matplotlib.cm.get_cmap('turbo')
        colors_lines = np.array(cmap_viridis([i/mid_point.shape[0] for i in range(mid_point.shape[0])]))

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        ax[0,0].scatter(pc[:, 0], pc[:, 1], color=TABLEAU_COLORS['tab:gray'])
        set_up_axes(ax[0,0])
        ax[0,0].quiver(mid_point[:, 0], mid_point[:, 1], line_vec_norm_orig[:, 0], line_vec_norm_orig[:, 1], 
                        color=colors_lines, edgecolor='k', width=0.01, linewidth=1)


        for i in range(mid_point.shape[0]):
            start_point = mid_point[i, :]
            line_vec = line_vec_norm_orig[i, :]
            end_point = start_point + line_vec * 1.0
            ax[0, 1].axline(start_point, xy2=end_point, c=colors_lines[i, :])

        set_up_axes(ax[0,1])
        ax[0, 1].set_xlim(*(ax[0,0].get_xlim()))
        ax[0, 1].set_ylim(*(ax[0,0].get_ylim()))


        # Plot Line Models in Parameter Space
        # polar_r = np.linalg.norm(condensed_param_set, axis=1)
        ax[0,2].remove()
        ax_polar = fig.add_subplot(2, 3, 3, projection='polar')
        ax_polar.scatter(np.radians(deg_ang), origin_offset, c=colors_lines, ec='k')
        # ax_polar.set_xlim(0, np.pi)
        # ax[0,1].scatter(parameter_set[:, 0], parameter_set[:, 1])
        ax[0,2].set_xlim(-1.175, 1.175)
        ax[0,2].set_xlabel(r"Angles (deg)")
        ax[0,2].set_ylabel(r"Origin Offset (m)")

    
        # Plot the Line Models as "points" in Cartesian Space (Euclidean Space)
        # Clusters are denoted by the marker shape
        # Color is still based upon the original line model.
        tab10_colors = np.array(plt.cm.get_cmap('tab20').colors)
        markers = ['o', 'v', '^', '<', '>', 'p', 's', 'P', '*', "X", "D", "d", "|", "_"]
        cluster_nums = np.unique(clusters)
        for i, cluster_num in enumerate(cluster_nums):
            mask = clusters == cluster_num
            marker = markers[i]
            colors = colors_lines[mask, :]
            ax[1,0].scatter(condensed_param_set[mask,0], condensed_param_set[mask,1], c=colors, ec='k', marker=marker)

        # Draw Ovals for the averaged clusters
        averaged_cluster_colors = dict()
        for cluster_num in cluster_idx:
            mask = clusters == cluster_num
            data = condensed_param_set[mask, :]
            average_color = np.mean(colors_lines[mask, :], axis=0)
            averaged_cluster_colors[cluster_num] = average_color
            bounding_elipse(data, ax[1,0], n_std=1.4, edgecolor='red')
            # confidence_ellipse(data[:, 0], data[:, 1], ax[1,0], n_std=3, edgecolor='red')
        # ax[1,0].scatter(condensed_param_set[:,0], condensed_param_set[:,1], c=tab10_colors[clusters], ec='k')
        set_up_axes(ax[1,0])

        # Plot Model Evaluation for one of the clusters
        # Show orthogonal distance offsets, mask out those that are outliers
        low_alpha = 0.3
        line_model = line_models_filtered[0]
        all_points = line_model['all_points']
        ortho_dir_dist = line_model['ortho_dir_dist'] * -1
        inlier_mask = line_model['inlier_mask']
        point_colors_alpha = np.ones((all_points.shape[0], 4))
        point_colors_alpha[:, :] = TABLEAU_COLORS['tab:gray']
        point_colors_alpha[~inlier_mask] = low_alpha
        ax[1, 1].scatter(pc[:, 0], pc[:, 1], c=point_colors_alpha, s=35)
        for i in range(all_points.shape[0]):
            point = all_points[i, :]
            new_point = point + ortho_dir_dist[i, :]
            line_color= np.array(TABLEAU_COLORS['tab:red'])
            line_color[3] = TABLEAU_COLORS['tab:red'][3] if inlier_mask[i] else low_alpha
            ax[1,1].plot([point[0], new_point[0]], [point[1], new_point[1]], color=line_color)
            point1 = line_model['line_point_old'] 
            point2 = line_model['line_point_old'] + line_model['dir_vec_old']
            ax[1,1].axline(point1, xy2=point2, c=averaged_cluster_colors[line_model['cluster_idx']], linestyle='--')
        set_up_axes(ax[1, 1])

        # Plot all the best models
        ax[1,2].scatter(pc[:, 0], pc[:, 1], color=TABLEAU_COLORS['tab:gray'])
        plot_fit_lines(ax[1,2], line_models_filtered, colors=averaged_cluster_colors)
        # plot_fit_lines(ax[1,1], line_models_filtered, colors=tab10_colors[np.array(cluster_idx)])
        set_up_axes(ax[1,2])



        plt.subplots_adjust(left=.124, bottom=.107, right=None, top=None, wspace=.267, hspace=.212)
        fig.savefig('assets/pics/lines_string_vectors.png', bbox_inches='tight')

        # Print *individual* images if needed
        for i, row in enumerate(ax):
            for j, ax_ in enumerate(row):
                if i == 0 and j == 2:
                    ax_ = ax_polar
                extent = ax_.get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
                pad = 0.05
                extent = extent.expanded(1.0 + pad, 1.0 + pad)
                fig.savefig(f'assets/pics/line_models_ax_{i}_{j}.png', bbox_inches=extent)
        
        plt.show()

        ms1 = (t2-t1) * 1000
        ms2 = (t3-t2) * 1000
        ms3 = (t4-t3) * 1000
        ms4 = (t5-t4) * 1000
        # ms5 = (t6-t5) * 1000
        # print(ms1, ms2, ms3, ms4)

    return line_models_filtered

def extract_lines_wrapper_new(top_points, top_normal, curb_height=0.5, **kwargs):
    t1 = time.perf_counter()
    top_points_3d = rotate_data_planar(top_points, top_normal)
    top_points_2d = top_points_3d[:, :2]
    height = np.mean(top_points_3d[:, 2])
    t2 = time.perf_counter()
    best_fit_lines = extract_lines_parameterized(top_points_2d, **kwargs)
    t3 = time.perf_counter()
    best_fit_lines = recover_3d_lines(best_fit_lines, top_normal, height, curb_height=curb_height)
    best_fit_lines = filter_lines(best_fit_lines, **kwargs)
    return best_fit_lines




def create_transform(translate, rotation):
    transform = np.eye(4)
    transform[:3, 3] = np.array(translate)
    if isinstance(rotation,np.ndarray) or isinstance(rotation, list):
        transform[:3, :3] = np.array(rotation)
    elif isinstance(rotation, dict):
        rot = R.from_euler('xyz', angles=[rotation['pitch'], rotation['roll'], rotation['yaw']], degrees=True)
        rm = rot.as_matrix()
        transform[:3, :3] = rm
    else:
        raise ValueError("Rotation must be an ndarray or a dictionary object with euler angles")
    return transform

def tranform_vectors(vectors, transform):
    point = np.array([0.0, 0.0, 0.0 , 1])
    point[:3] = vectors
    new_point = (transform @ point)[:3]
    return new_point

def rotate_vectors(vectors, transform):
    rot = transform[:3, :3]
    new_vec = rot @ vectors
    return new_vec

def transform_points(points, transform):
    """
    Transform points using a given 4x4 transformation matrix
    """
    points_ = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed = np.ascontiguousarray(np.matmul(points_, transform.transpose())[:, :3])
    return points_transformed

def get_turning_manuever(platform_center_sensor_frame, platform_normal_sensor_frame, sensor_to_wheel_chair_transform, **kwargs):
    """Will compute the turning manuever for the wheelchair. CC = Counter Clockwise

    Args:
        platform_center_sensor_frame (np.ndarray[3,]): The platform center point in the SENSOR FRAME
        platform_normal_sensor_frame (np.ndarray[3,]): The platform normal in the SENSOR FRAME. Normal is pointing TOWARDS the wheelchair.
        sensor_to_wheel_chair_transform (np.ndarray[4X4]): The homogenous transform to transfrom from the SENSOR FRAME to the wheelchair FRAME

    Returns:
        dict: A dictionary of the maneuver
            alpha = angle between wheelchair y-axis (forward) and vector from wheelchair to point of interest
            beta = angle between wheelchair y-axis (forward) and reversed platform normal
            dist_poi = 2D distance between wheelchair center and point of interest
            ortho_dist_platform = 2D orthogonal distance between wheelchair center and center of platform
            first_turn = angle to turn wheelchair to align with vector directing wheelchair to point of interst (alpha)
            second_turn = angle to turn wheelchair to align with platform normal (-alpha + beta)
            vec_wheel_chair_to_poi_2D_unit = 2D unit vector from wheelchair to poi (wheelchair frame)
            platform_normal_inverted_unit = 3D unit vector of the inverse of the platform normal 
    """
    platform_center_pos_wheel_chair_frame = tranform_vectors(platform_center_sensor_frame, sensor_to_wheel_chair_transform)
    platform_normal_wheel_chair_frame = rotate_vectors(platform_normal_sensor_frame, sensor_to_wheel_chair_transform)

    result = compute_turning_manuever(platform_center_pos_wheel_chair_frame, platform_normal_wheel_chair_frame, **kwargs)

    return result

def compute_2D_angle_difference(vector1, vector2):
    """Computes the 2D angle difference between vector 1 and vector 2
    The sign of the angle is the CC rotation to move vector1 TO vector2
    Args:
        vector1 (np.ndarray): vector 1
        vector2 (np.ndarray): vector 2

    Returns:
        float: Angle difference in degrees
    """
    angle = np.degrees(np.arctan2(vector2[1], vector2[0]) - np.arctan2(vector1[1], vector1[0]))

    if angle > 180:
        angle = angle - 360 
    elif angle < -180:
        angle = angle + 360

    return angle

def compute_turning_manuever(platform_center_pos_wheel_chair_frame, platform_normal_wheel_chair_frame, best_fit_line, poi_offset=0.5, debug=False, **kwargs):
    """Will compute the turning manuever for the wheelchair. CC = Counter Clockwise
    Assumes that the wheelchair reference frame origin is the center of rotation for wheelchair turn commands
    Maneuvers Steps:
      0. Calculate point of interest (POI) as `platform_normal_wheel_chair_frame * poi_offset + platform_center_pos_wheel_chair_frame`
      1. Execute rotation of "first_turn" degrees CC. This aligns the wheelchair y-axis to point to the POI
      2. Execute forward move (y-axis) "dist_poi" meters
      3. Execute rotation of "second_turn" degrees CC. This aligns the wheelchair y-axis to platform

    This assumes perfect control and execution of these commands. In reality you will need feeback control probably for 
    steps 2 and 3. Step 1 is 'probably' not as important to need feedback control.

    Args:
        platform_center_pos_wheel_chair_frame (np.ndarray): The platform center point in the wheelchair FRAME
        platform_normal_wheel_chair_frame (np.ndarray): The platform normal in the wheelchair FRAME. Normal is pointing TOWARDS the wheelchair.
        debug (bool, optional): Whether to print out data and plot. Defaults to False.

    Returns:
        dict: A dictionary of the maneuver
            alpha = angle between wheelchair y-axis (forward) and vector from wheelchair to point of interest
            beta = angle between wheelchair y-axis (forward) and reversed platform normal
            dist_poi = 2D distance between wheelchair center and point of interest
            ortho_dist_platform = 2D orthogonal distance between wheelchair center and center of platform
            first_turn = angle to turn wheelchair to align with vector directing wheelchair to point of interst (alpha)
            second_turn = angle to turn wheelchair to align with platform normal (-alpha + beta)
            vec_wheel_chair_to_poi_2D_unit = 2D unit vector from wheelchair to poi (wheelchair frame)
            platform_normal_inverted_unit = 3D unit vector of the inverse of the platform normal 
    """
    platform_poi_pos_wheel_chair_frame = platform_normal_wheel_chair_frame * poi_offset + platform_center_pos_wheel_chair_frame
    wheel_chair_pos_in_wheel_chair_frame = np.array([0.0,0.0,0.0]) # The position will be 0 in wheelchair frame (origin)
    wheel_chair_dir_vec_unit = np.array([0.0, 1.0, 0.0]) # forward y-axis is wheelchair direction

    # Orthogonal distance to the platform
    ortho_dist_platform = np.dot(platform_center_pos_wheel_chair_frame, platform_normal_wheel_chair_frame)
    ortho_dist_platform = np.abs(ortho_dist_platform)

    vec_wheelchair_to_poi = platform_poi_pos_wheel_chair_frame - wheel_chair_pos_in_wheel_chair_frame # called Vec3 (blue) in diagrams
    vec_wheel_chair_to_poi_2D = vec_wheelchair_to_poi[:2] # the z-axis is height in this reference frame
    dist_poi = np.linalg.norm(vec_wheel_chair_to_poi_2D)
    vec_wheel_chair_to_poi_2D_unit = vec_wheel_chair_to_poi_2D / dist_poi

    platform_normal_inverted_unit = -platform_normal_wheel_chair_frame # called Vec2 (red) in diagrams

    alpha = compute_2D_angle_difference(wheel_chair_dir_vec_unit, vec_wheel_chair_to_poi_2D_unit)
    beta = compute_2D_angle_difference(wheel_chair_dir_vec_unit,platform_normal_inverted_unit )

    first_turn = alpha
    second_turn = -alpha + beta

    result = dict(alpha=alpha, beta=beta, dist_poi=dist_poi, ortho_dist_platform=ortho_dist_platform,first_turn=first_turn, second_turn=second_turn, 
                vec_wheel_chair_to_poi_2D_unit=vec_wheel_chair_to_poi_2D_unit, platform_normal_inverted_unit=platform_normal_inverted_unit,
                platform_center_pos_wheel_chair_frame=platform_center_pos_wheel_chair_frame, platform_normal_wheel_chair_frame=platform_normal_wheel_chair_frame,
                platform_poi_pos_wheel_chair_frame=platform_poi_pos_wheel_chair_frame, poi_offset=poi_offset)
    if debug:
        print(f"Alpha Angle {alpha:.1f}; Beta Angle: {beta:.1f}")
        print(f"First Turn CC: {first_turn:.1f} degrees; Move Distance: {dist_poi:.2f}; Second Turn: {second_turn:.1f}")
        plot_maneuver(result, best_fit_line)

    return result

def plot_maneuver(result, best_fit_line):

    platform_poi_pos_wheel_chair = result['platform_poi_pos_wheel_chair_frame']
    platform_center = result['platform_center_pos_wheel_chair_frame']
    platform_square = best_fit_line['square_points']

    # Manually scale platform normal to reach POI
    fig, ax = plt.subplots(1, 1)
    # Plot Platform center
    ax.scatter(platform_center[0], platform_center[1], c='k', zorder=3)
    # Plot Platform Line (square box projected to XY plane)
    ax.plot(platform_square[:2, 0], platform_square[:2, 1], c=[1.0, 0.0, 0.0, 1.0])
    ax.text(platform_center[0] - 0.25, platform_center[1], 'Curb')
    # Plot POI
    ax.scatter(platform_poi_pos_wheel_chair[0], platform_poi_pos_wheel_chair[1], c='tab:blue', ec='k', zorder=5)
    ax.text(platform_poi_pos_wheel_chair[0] - 0.18, platform_poi_pos_wheel_chair[1]-0.01, 'POI', zorder=5)
    # Plot Wheel chair origin
    ax.scatter(0, 0, c='k', zorder=4)
    ax.text(0.05, -0.05, 'Wheelchair Origin')

    # Plot Platform Normal (red)
    platform_normal = -0.62 *result['platform_normal_inverted_unit']
    arrow_platform_normal = arrow_(ax, platform_center[0], platform_center[1], platform_normal[0], platform_normal[1], ec='tab:red', fc='tab:red', width=.01) #label=rf'Platform Normal, $\boldsymbol{$\alpha$}$'
    # Plot Wheel Chair Direction (green)
    arrow_wc_dir = arrow_(ax, 0.0, 0.0, 0, 1.0, ec='tab:green', fc='tab:green', width=.01)
    # Plot Direction to POI from Wheel Chair Start (blue)
    arrow_poi_dir = arrow_(ax, 0.0, 0.0, result['vec_wheel_chair_to_poi_2D_unit'][0], result['vec_wheel_chair_to_poi_2D_unit'][1], ec='tab:blue', fc='tab:blue', width=.01)
    # Plot Platform Inverted Normal (red)
    arrow_normal_inv = arrow_(ax,0.0, 0.0, result['platform_normal_inverted_unit'][0], result['platform_normal_inverted_unit'][1], ec='tab:purple', fc='tab:purple', width=.01)

    # Angle Annotations
    AngleAnnotation((0.0, 0.0),[0, 1], result['vec_wheel_chair_to_poi_2D_unit'], ax=ax, fig=fig, size=125, text=rf'$\alpha$', textposition='outside') # text_kw=dict(bbox=dict(boxstyle="round", fc="w"))
    AngleAnnotation((0.0, 0.0),result['platform_normal_inverted_unit'][:2], [0,1], ax=ax, fig=fig, size=175, text=rf'$\beta$', textposition='outside')

    # Create Legend, need custom code to draw arrows
    import matplotlib.patches as mpatches
    from matplotlib.legend_handler import HandlerPatch
    def make_legend_arrow(legend, orig_handle,
                        xdescent, ydescent,
                        width, height, fontsize):
        p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
        return p
    # Platform normal vector
    plt.legend([arrow_platform_normal, arrow_wc_dir, arrow_poi_dir, arrow_normal_inv], 
            [r'Curb Normal, $\mathbf{n_c}$',
             r'Wheelchair Direction, $\mathbf{v_1}$',
             r'POI Direction, $\mathbf{v_2}$',
             r'Inv Curb Normal, $-\mathbf{n_c}$'
             ], loc='upper right', fontsize=11,
             handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),
                    })

    ax.axis('equal')

    fig.savefig('assets/pics/plot_maneuver.png', bbox_inches='tight')


def arrow_(ax, x, y, dx, dy, **kwargs):
    arrow = ax.arrow(x, y, dx, dy, **kwargs)
    ax.scatter(x + dx + 0.1, y + dy + 0.1, alpha=0.0)
    return arrow
def get_theta_and_distance(plane_normal, point_on_plane, ground_normal):
    """
    This code calculates the angle and distance to the curb

    'point_on_plane' = Centroid of red box
    'plane_normal' or 'normal' = Normal of the red box
    'ground_normal' = Normal of the ground plane
    All data is in the reference frame of the camera

    """

    print(plane_normal, ground_normal)
    # Currently the Cameras are placed on the right hand side of the wheelchair
    # The origin of the cameras need to be offsetted
    # so that it takes in account to be in the center of the wheelchair and the footrest
    wheelchair_center = np.array([-0.3, 0, 0])

    """
    Point of interest is a point that is 0.5 meter away from the center of the plane 
    Calculation:
    point_of_interest = point_on_plane + (distance [0.5 meters] * unit vector) 

    """
    point_on_plane_unit_vector = plane_normal / np.linalg.norm(plane_normal)
    point_of_interest = point_on_plane + (0.7 * point_on_plane_unit_vector) # check the assumption
    # threshold = np.linalg.norm(point_of_interest - point_on_plane)
    # import ipdb; ipdb.set_trace()
    
    # Orthogonal Distance to the hyerplane to the curb
    diff = wheelchair_center - point_on_plane
    orthog_dist = np.dot(diff, plane_normal)
    orthog_dist = np.abs(orthog_dist)

    # Orientation to the Curb
    vectors = np.array([[0.0, 0.0, -1.0], plane_normal]) # Plane normal is 
    # To project onto ground plane
    vectors_proj = project_points_geometric_plane(
        vectors, ground_normal, np.array([0.0, 0.0, 0.0]))
    # Vector 1 = 2D vector of the wheelchair forward position   #####
    # import ipdb; ipdb.set_trace()
    vec1 = vectors_proj[0, :]
    vec1 = vec1 / np.linalg.norm(vec1)
    # Vector 2 = 2D vector of the red box normal
    vec2 = vectors_proj[1, :]
    vec2 = vec2 / np.linalg.norm(vec2)
    a = np.dot(vec1, vec2)
    # orthog_ang = Are you parralel to the curb?
    orientation = np.degrees(np.arccos(a))
    # Tell you if you are left or right to the curb
    cross = np.cross(vec1, vec2)
    if (np.dot(ground_normal, cross) < 0):
        orientation = -orientation

    # import ipdb; ipdb.set_trace()
    
    # FINAL TURN Calculations
    
    # Coordinates of the wheelchair center and point of interest
    # points = np.array([wheelchair_center, point_of_interest])
    vec3_3d = point_of_interest - np.array([0.0,0.0,0.0])

    # Coordinates projected with respect to the ground plane
    vector_proj = project_points_geometric_plane([vec3_3d], ground_normal, np.array([0.0, 0.0, 0.0]))
    vec3_2d = vector_proj[0, :]
    vec3_2d = vec3_2d / np.linalg.norm(vec3_2d)

    vec4_2d = vec2

    dot_prod = np.dot(vec3_2d, vec4_2d)
    final_turn = np.degrees(np.arccos(dot_prod))
    # wheel_chair_2d = points_proj[0, :]
    # poi_2d = points_proj[1, :]

    # vec3_2d = poi_2d - wheel_chair_2d
    # vec3_2d = vec3_2d / np.linalg.norm(vec3_2d)

    # vec4_2d = vec2 * -1

    # dot_prod = np.dot(vec3_2d, vec4_2d)
    # final_turn = np.degrees(np.arccos(dot_prod))

    # TODO Checks on wheelchair position (line test) in relation to poi and plane normal
    


    points2 = np.array([wheelchair_center, point_of_interest])
    points_proj2 = project_points_geometric_plane(points2, ground_normal, np.array([0.0, 0.0, 0.0]))
    # Calculate distance and angle between wheelchair center and point of interest
    distance_of_interest = np.linalg.norm(points_proj2[0, :] - points_proj2[1, :])
    # final_turn = np.degrees(np.arccos((orthog_dist - 0.5)/distance_of_interest))
    # final_turn = (math.acos((orthog_dist - 0.5) / distance_of_interest)) * 180/math.pi

    # vec3 = points_proj[0, :]
    # vec3 = vec3 / np.linalg.norm(vec3)
    # a2 = np.dot(vec2, vec3)
    # final_turn = np.degrees(np.arccos(a2))
    # cross2 = np.cross(vec2, vec3)
    # if (np.dot(ground_normal, cross2) > 0):
    #     final_turn = -final_turn

    
    # INITIAL TURN
    # a3 = np.dot(vec1, vec3)
    # initial_turn = np.degrees(np.arccos(a3))
    # cross3 = np.cross(vec1, vec3)
    # if (np.dot(ground_normal, cross3) > 0):
    #     initial_turn = -initial_turn

    # import ipdb; ipdb.set_trace()

    return orthog_dist, distance_of_interest, final_turn, orientation
