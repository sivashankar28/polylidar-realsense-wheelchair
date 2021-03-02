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
import logging
from itertools import combinations

import numpy as np
from scipy.ndimage.filters import uniform_filter1d
from scipy.spatial.transform import Rotation as R
from simplifyline import MatrixDouble, simplify_radial_dist_3d
from surfacedetector.utility.helper_general import rotate_data_planar, normalized
import matplotlib.pyplot as plt


def choose_plane(first_plane, second_plane):
    first_centroid = first_plane['point']
    second_centroid = second_plane['point']
    first_normal = first_plane['normal_ransac']

    proj_first = np.dot(first_centroid, first_normal)
    proj_second = np.dot(second_centroid, first_normal)

    if proj_second < proj_first:
        return second_plane
    else:
        return first_plane


def filter_points(top_points, max_z=0.5, max_dist=0.05):
    """Filters and simplifies 3D points belonging to a continuos line string.


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
    a1 = top_points_simplified[:, 2] < far_z

    np_diff = np.diff(np.hstack(([False], a1 == True, [False])))
    idx_pairs = np.where(np_diff)[0].reshape(-1, 2)
    great_idx = np.diff(idx_pairs, axis=1).argmax()
    start_idx, end_idx = idx_pairs[great_idx, 0], idx_pairs[great_idx, 1]

    filtered_top_points = top_points_simplified[start_idx:end_idx, :]
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


def fit_line(points, idx, max_slope=2.0):
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
               rmse=rmse, dir_vec=dir_vec, idx=[idx[0], last_idx], flip_axis=flip_axis)
    logging.debug("Fit a new line: %s", res)
    return res


def extract_lines(pc, window_size=3, dot_min=0.90, **kwargs):
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


def orthogonal_distance(line_point, line_vec, points):
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
    _, lengths = normalized(perf_offset)
    median = np.median(lengths)

    return median


def check_merge_line(points, line, line_next, i, max_idx_dist=3, max_rmse=1.0, min_dot_prod=0.93, max_ortho_dist=0.05):
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


def merge_lines(points, lines, max_idx_dist=5, max_rmse=1.0, min_dot_prod=0.90, max_ortho_dist=0.05, **kwargs):
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


def filter_lines(best_fit_lines, max_dot=0.2, w1=0.75, w2=0.25, return_only_one_line=True, **kwargs):
    """Filter lines to a max of 2, they must be orthogonal
    If multiple pairs can be found, find the one that maximizes:
    metric = w1 * (1- dot_prod) * w2 * idx_length
    Basically the pair of line strings that are the longest and most orthogonal

    Args:
        best_fit_lines (List): List of Lines
        max_dot (float, optional): Max dot product if more than two lines. Defaults to 0.2.
        w1 (float, optional): Weight of dot product. Defaults to 0.75.
        w2 (float, optional): Weight of Line Lenght. Defaults to 0.25

    Returns:
        [type]: [description]
    """
    best_pair = []
    if return_only_one_line and best_fit_lines:
        # only return one line, choose the closest line
        # TODO RMSE and Length metric as well
        for line in best_fit_lines:
            line['distance'] = np.linalg.norm(line['hplane_point'])
        return sorted(best_fit_lines, key=lambda i: i['distance'])[:1]
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
    for line in best_fit_lines:
        line['distance'] = np.linalg.norm(line['hplane_point'])

    return sorted(best_pair, key=lambda i: i['distance'])


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


def recover_3d_lines(best_fit_lines, top_normal, height):
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
        line['points_3d'] = np.append(
            pts, np.ones((pts.shape[0], 1)) * height, axis=1)
        line['points_3d'] = rotate_data_planar(
            line['points_3d'], top_normal, True)
        line['dir_vec_3d'] = np.array(
            [[line['dir_vec'][0], line['dir_vec'][1], 0]])
        line['dir_vec_3d'] = rotate_data_planar(
            line['dir_vec_3d'], top_normal, True).flatten()
        line['dir_vec_3d'] = line['dir_vec_3d'] / \
            np.linalg.norm(line['dir_vec_3d'])
        line['plane_normal'] = top_normal
        line['hplane_normal'] = np.cross(
            line['dir_vec_3d'], line['plane_normal']) * -1
        line['hplane_normal'] = line['hplane_normal'] / \
            np.linalg.norm(line['hplane_normal'])
        line['hplane_point'] = line['points_3d'].mean(axis=0)
        line['square_points'] = make_square(
            line['hplane_point'], line['plane_normal'], line['dir_vec_3d'], line['hplane_normal'])

    return best_fit_lines


def extract_lines_wrapper(top_points, top_normal, min_points_line=12, **kwargs):
    t1 = time.perf_counter()
    top_points_3d = rotate_data_planar(top_points, top_normal)
    top_points_2d = top_points_3d[:, :2]
    height = np.mean(top_points_3d[:, 2])
    t2 = time.perf_counter()
    all_fit_lines = extract_lines(top_points_2d)
    t3 = time.perf_counter()
    best_fit_lines = merge_lines(top_points_2d, all_fit_lines)
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

def get_turning_manuever(platform_center_sensor_frame, platform_normal_sensor_frame, sensor_to_wheel_chair_transform, **kwargs):
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
    angle = np.arctan2(vector2[1], vector2[0]) - np.arctan2(vector1[1], vector1[0])

    if angle > 180:
        angle = angle - 360
    elif angle < -180:
        angle = angle + 360

    return np.degrees(angle)

def compute_turning_manuever(platform_center_pos_wheel_chair_frame, platform_normal_wheel_chair_frame, poi_offset=0.5, debug=False, **kwargs):
    """Will compute the turning manuever for the wheel chair. CC = Counter Clockwise
    Assumes that the Wheel Chair reference frame origin is the center of rotation for wheel chair turn commands
    Maneuvers Steps:
      0. Calculate point of interest (POI) as `platform_normal_wheel_chair_frame * poi_offset + platform_center_pos_wheel_chair_frame`
      1. Execute rotation of "first_turn" degrees CC. This aligns the wheel chair y-axis to point to the POI
      2. Execute forward move (y-axis) "dist" meters
      3. Execute rotation of "second_turn" degrees CC. This aligns the wheel chair y-axis to platform

    This assumes perfect control and execution of these commands. In reality you will need feeback control probably for 
    steps 2 and 3. Step 1 is 'probably' not as important to need feedback control.

    Args:
        platform_center_pos_wheel_chair_frame (np.ndarray): The platform center point in the WHEEL CHAIR FRAME
        platform_normal_wheel_chair_frame (np.ndarray): The platform normal in the WHEEL CHAIR FRAME. Normal is pointing TOWARDS the wheel chair.
        debug (bool, optional): Whether to print out data. Defaults to False.

    Returns:
        dict: A dictionary of the maneuver
            alpha = angle between wheel chair y-axis (forward) and vector from wheel chair to point of interest
            beta = angle between wheel chair y-axis (forward) and reversed platform normal
            dist_poi = 2D distance between wheel chair center and point of interest
            ortho_dist_platform = 2D orthogonal distance between wheel chair center and center of platform
            first_turn = angle to turn wheel chair to align with vector directing wheel chair to point of interst (alpha)
            second_turn = angle to turn wheel chair to align with platform normal (-alpha + beta)
            vec_wheel_chair_to_poi_2D_unit = 2D unit vector from wheel chair to poi (wheel chair frame)
            platform_normal_inverted_unit = 3D unit vector of the inverse of the platform normal 
    """
    platform_poi_pos_wheel_chair_frame = platform_normal_wheel_chair_frame * poi_offset + platform_center_pos_wheel_chair_frame
    wheel_chair_pos_in_wheel_chair_frame = np.array([0.0,0.0,0.0]) # The position will be 0 in wheel chair frame (origin)
    wheel_chair_dir_vec_unit = np.array([0.0, 1.0, 0.0]) # forward y-axis is wheel chair direction

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
                platform_poi_pos_wheel_chair_frame=platform_poi_pos_wheel_chair_frame)
    if debug:
        print(f"Alpha Angle {alpha:.1f}; Beta Angle: {beta:.1f}")
        print(f"First Turn CC: {first_turn:.1f} degres; Move Distance: {dist_poi:.2f};  Second Turn: {second_turn:.1f}")
        plot_maneuver(result)

    return result

def plot_maneuver(result):
    platform_poi_pos_wheel_chair = result['platform_poi_pos_wheel_chair_frame']
    platform_center = result['platform_center_pos_wheel_chair_frame']

    fig, ax = plt.subplots(1, 1)
    ax.scatter(platform_center[0], platform_center[1], c=[[1, 0, 0]])
    ax.text(platform_center[0] + 0.03, platform_center[1], 'platform')
    ax.scatter(platform_poi_pos_wheel_chair[0], platform_poi_pos_wheel_chair[1], c=[[0, 1, 0]])
    ax.text(platform_poi_pos_wheel_chair[0] + 0.03, platform_poi_pos_wheel_chair[1], 'poi')
    ax.scatter(0, 0, c='k')
    ax.text(0.01, 0, 'Wheel Chair')
    arrow_(ax, 0.0, 0.0, 0, 1, ec='g', fc='g', width=.01)
    arrow_(ax, 0.0, 0.0, result['vec_wheel_chair_to_poi_2D_unit'][0], result['vec_wheel_chair_to_poi_2D_unit'][1], ec='b', fc='b', width=.01)
    arrow_(ax,0.0, 0.0, result['platform_normal_inverted_unit'][0], result['platform_normal_inverted_unit'][1], ec='r', fc='r', width=.01)
    ax.text(result['vec_wheel_chair_to_poi_2D_unit'][0] / 2.0, (result['vec_wheel_chair_to_poi_2D_unit'][1] + 1) / 2.0, rf'$\alpha={result["alpha"]:.0f}^\circ$')
    ax.text((0.0 + result['platform_normal_inverted_unit'][0]) / 2.0, 
        (1.0 + result['platform_normal_inverted_unit'][1]) / 2.0, rf'$\beta={result["beta"]:.0f}^\circ$')
    ax.axis('equal')

def arrow_(ax, x, y, dx, dy, **kwargs):
    ax.arrow(x, y, dx, dy, **kwargs)
    ax.scatter(x + dx + 0.1, y + dy + 0.1, alpha=0.0)

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
    point_of_interest = point_on_plane + (0.5 * point_on_plane_unit_vector) # check the assumption
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

    # TODO Checks on wheel chair position (line test) in relation to poi and plane normal
    


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
