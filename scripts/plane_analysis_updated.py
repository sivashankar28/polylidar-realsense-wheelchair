from scipy.spatial.transform import Rotation as R
from joblib import load
import numpy as np
import matplotlib.pyplot as plt
from surfacedetector.utility.helper_general import set_axes_equal, plot_points, setup_figure_2d, setup_figure_3d
from pathlib import Path
import logging
import time


from surfacedetector.utility.helper_linefitting import extract_lines_wrapper, filter_points, choose_plane

logging.basicConfig(level=logging.INFO)


DATA_DIR = Path('./data/scratch_test')


def get_files():
    p = DATA_DIR.glob('*curbmodel_*')
    files = sorted([x for x in p if x.is_file()])
    return files


def visualize_3d(first_points_rot, second_points_rot=None, line_1=None):
    fig, ax = setup_figure_3d()
    plot_points(ax, first_points_rot)
    if second_points_rot is not None:
        plot_points(ax, second_points_rot)
    if line_1 is not None:
        next(ax._get_lines.prop_cycler)
        plot_points(ax, line_1, is_line=True, linewidth=7)
    ax.set_box_aspect([1, 1, 1])  # IMPORTANT - this is the new, key line
    set_axes_equal(ax)  # IMPORTANT - this is also required
    # ax.invert_zaxis()
    plt.show()



def plot_fit_lines(ax, fit_lines):
    next(ax._get_lines.prop_cycler)
    for fit_line in fit_lines:
        poly1d_fn = fit_line['fn']
        if fit_line['flip_axis']:
            ax.plot(poly1d_fn(fit_line['x_points']), fit_line['x_points'], '-')
        else:
            points = fit_line['points']
            ax.plot(points[:, 0], poly1d_fn(points[:, 0]), '-')
        mean = fit_line['points'].mean(axis=0)
        ax.annotate(f"RMSE={fit_line['rmse']:.3f}", (mean[0], mean[1]))



def visualize_2d(top_points, top_normal, min_points_line=6):
    top_points_2d, height, all_fit_lines, best_fit_lines = extract_lines_wrapper(
        top_points, top_normal, min_points_line)
    fig, ax = setup_figure_2d()
    plot_points(ax[0], top_points_2d)
    for i in range(top_points_2d.shape[0]):
        ax[0].annotate(str(i), (top_points_2d[i, 0], top_points_2d[i, 1]))
    plot_fit_lines(ax[0], all_fit_lines)
    plot_fit_lines(ax[1], best_fit_lines)
    plt.show()

def process(data):
    """ Process the bottom and top planes dictionary """

    top_plane = choose_plane(data['first_plane'], data['second_plane'])
    top_points, top_normal = top_plane['all_points'], top_plane['normal_ransac']
    # visualize_3d(top_points)
    t1 = time.perf_counter()
    filtered_top_points = filter_points(top_points)  # <100 us
    t2 = time.perf_counter()
    visualize_2d(filtered_top_points, top_normal)
    _, height, _, best_fit_lines = extract_lines_wrapper(
        filtered_top_points, top_normal)  # ~2ms
    t3 = time.perf_counter()
    ms1 = (t2-t1) * 1000
    ms2 = (t3-t2) * 1000
    ms = ms1 + ms2
    logging.debug(
        "Process Points - Filter and Simplify: %.2f, Extract Lines: %.2f", ms1, ms2)
    # visualize_3d(top_points, second_points_rot=best_fit_lines[0]['square_points'], line_1=best_fit_lines[0]['points_3d'])
    return ms


def main():
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


if __name__ == "__main__":
    main()
