import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import warnings

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