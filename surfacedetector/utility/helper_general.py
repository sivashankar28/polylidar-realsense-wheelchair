import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import warnings


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

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


def rotate_data_planar(points, normal, inverse=False):
    if normal is None or np.array_equal(normal, [0.0, 0.0, 1.0]):
        return points
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rm, _ = R.align_vectors([[0, 0, 1]], [normal])  # Rotating matrix the polygon
    return rm.apply(points, inverse=inverse)


def plot_points(ax, X, c=None, is_line=False, linewidth=2):
    a = [a.flatten() for a in np.split(X, X.shape[1], axis=1)]
    if c is not None:
        if is_line:
            ax.plot(*a, c=c, linewidth=linewidth)
        else:
            ax.scatter(*a, c=c)
    else:
        if is_line:
            ax.plot(*a, linewidth=linewidth)
        else:
            ax.scatter(*a)

def setup_figure_3d():
    fig, ax_ = plt.subplots(nrows=1, ncols=1, subplot_kw=dict(projection='3d'))
    ax_.set_xlabel("X")
    ax_.set_ylabel("Y")
    ax_.set_zlabel("Z")

    return fig, ax_

def setup_figure_2d():
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(17, 5))
    for ax_ in ax:
        ax_.set_xlabel("X")
        ax_.set_ylabel("Y")
        ax_.axis('equal')
    return fig, ax

def plot_fit_lines(ax, fit_lines, annotate=True, colors=None):
    next(ax._get_lines.prop_cycler)
    for i, fit_line in enumerate(fit_lines):
        poly1d_fn = fit_line['fn']
        if colors is None:
            color = next(ax._get_lines.prop_cycler)['color']
        else:
            color = colors[i]
        if fit_line['flip_axis']:
            ax.plot(poly1d_fn(fit_line['x_points']), fit_line['x_points'], '-',  c=color)
        else:
            points = fit_line['points']
            ax.plot(points[:, 0], poly1d_fn(points[:, 0]), '-', c=color)
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