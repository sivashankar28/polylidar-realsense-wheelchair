import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import warnings

def create_transform(translate=[0.0, 0.0], rot=45, deg=True):
    transform = np.eye(3)
    rad = np.radians(rot) if deg else rot
    rm = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)]
    ])

    transform[:2,:2] = rm
    transform[:2, 2] = translate
    return transform

def draw_brace_updated(ax, p1, p2, text, rotation=0, zorder=1):
    """Draws an annotated brace on the axes."""
    ax.autoscale(False)

    p1 = np.array(p1)
    p2 = np.array(p2)
    # Get Direction vector of two points
    vec = np.array([p2[0] - p1[1], p2[1]- p1[1]])
    # Get brace length
    brace_length = np.linalg.norm(vec)
    # normalize vec
    vec = vec / brace_length
    
    # Get angle of rotation
    dot = 1.0 * vec[0]
    det = 1.0 * vec[1]
    rotation = np.degrees(np.arctan2(det, dot))

    # Get the mean point, set height level
    mean_point = (p1 + p2) / 2.0
    yy = mean_point[1]

    # calculate how much the non-roated brace has to expand
    # such that it spans the correct length once rotated
    x_shrink = np.cos(np.radians(rotation))
    x_scale = (1.0 / x_shrink) * .90
    brace_length_flat = brace_length * x_scale
    to_add = (brace_length_flat - brace_length) / 2.0
    xmin, xmax = p1[0] -to_add, p2[0] + to_add


    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy + (.05*y - .01)*yspan # adjust vertical position

    p_idx_text = np.argmax(y)
    # print(p_idx_text)
    # print(x[p_idx_text], y[p_idx_text])

    x_min = np.mean(x)
    y_min = np.min(y)
    y_max = np.max(y)
    # print(xspan, yy)
    # print(x_min, y_min)

    center = [x_min, y_min]

    points = np.column_stack([x, y])
    points_1 = np.ones((3, points.shape[0]))
    points_1[:2,:] = (points - center).T

    rm = create_transform(translate=center, rot=rotation)
    points_rot = rm @ points_1
    x = points_rot[0, :]
    y = points_rot[1, :]

    text_center = points_rot[:2, p_idx_text].T

    dir_vec = (rm @ [[0], [1.0], [1.0]])[:2, 0].T
    # print(dir_vec)
    # print(text_center)

    alpha = (y_max - y_min) / 2.0
    text_center = text_center + dir_vec * alpha

    ax.plot(x, y, color='black', lw=1, zorder=zorder)
    ax.text(text_center[0], text_center[1], text, ha='center', va='center', rotation=rotation)

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


def get_angle(vec):
    dot = 1.0 * vec[0]
    det = 1.0 * vec[1]
    rotation = np.degrees(np.arctan2(det, dot))
    return rotation

def plot_fit_lines(ax, fit_lines, annotate=True, colors=None):
    next(ax._get_lines.prop_cycler)
    for i, fit_line in enumerate(fit_lines):
        poly1d_fn = fit_line['fn']
        if colors is None:
            color = next(ax._get_lines.prop_cycler)['color']
        else:
            if isinstance(colors, dict):
                color = colors[fit_line['cluster_idx']]
            else:
                color = colors[i]
        if fit_line['flip_axis']:
            ax.plot(poly1d_fn(fit_line['x_points']), fit_line['x_points'], '-',  c=color)
        else:
            points = fit_line['points']
            ax.plot(points[:, 0], poly1d_fn(points[:, 0]), '-', c=color)
        mean = fit_line['points'].mean(axis=0)
        if annotate:
            ang = get_angle(fit_line['dir_vec'])
            rm = create_transform(rot=ang)
            dir_vec = (rm @ [[0], [1.0], [1.0]])[:2, 0].T
            text_center = mean + dir_vec * 0.1
            # print(fit_line['dir_vec'], ang)
            ax.annotate(f"RMSE={fit_line['rmse']:.3f}", (text_center[0], text_center[1]), rotation=ang, ha='center', va='center', )

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