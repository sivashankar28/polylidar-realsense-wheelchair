import time
import logging
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# from polylidar.polylidarutil.plane_filtering import filter_planes_and_holes
from polylidar import MatrixDouble, Polylidar3D
from polylidar.polylidarutil.open3d_util import create_lines

from fastgac import GaussianAccumulatorS2Beta, MatX3d, IcoCharts
from fastgac.peak_and_cluster import find_peaks_from_ico_charts
from fastgac.o3d_util import get_arrow, get_pc_all_peaks, get_arrow_normals

from surfacedetector.utility.helper_wheelchair import extract_geometric_plane
from surfacedetector.utility.helper_planefiltering import filter_planes_and_holes

# from surfacedetector.helper_mesh import create_open_3d_mesh

import open3d as o3d


# def split_triangles(mesh):
#     """
#     Split the mesh in independent triangles
#     """
#     triangles = np.asarray(mesh.triangles).copy()
#     vertices = np.asarray(mesh.vertices).copy()

#     triangles_3 = np.zeros_like(triangles)
#     vertices_3 = np.zeros((len(triangles) * 3, 3), dtype=vertices.dtype)

#     for index_triangle, t in enumerate(triangles):
#         index_vertex = index_triangle * 3
#         vertices_3[index_vertex] = vertices[t[0]]
#         vertices_3[index_vertex + 1] = vertices[t[1]]
#         vertices_3[index_vertex + 2] = vertices[t[2]]

#         triangles_3[index_triangle] = np.arange(index_vertex, index_vertex + 3)

#     mesh_return = deepcopy(mesh)
#     mesh_return.triangles = o3d.utility.Vector3iVector(triangles_3)
#     mesh_return.vertices = o3d.utility.Vector3dVector(vertices_3)
#     mesh_return.triangle_normals = mesh.triangle_normals
#     mesh_return.paint_uniform_color([0.5, 0.5, 0.5])
#     return mesh_return


# def assign_some_vertex_colors(mesh, triangle_indices, triangle_colors, mask=None):
#     """Assigns vertex colors by given normal colors
#     NOTE: New mesh is returned

#     Arguments:
#         mesh {o3d:TriangleMesh} -- Mesh
#         normal_colors {ndarray} -- Normals Colors

#     Returns:
#         o3d:TriangleMesh -- New Mesh with painted colors
#     """
#     split_mesh = split_triangles(mesh)
#     vertex_colors = np.asarray(split_mesh.vertex_colors)
#     triangles = np.asarray(split_mesh.triangles)
#     if mask is not None:
#         triangles = triangles[mask, :]

#     if isinstance(triangle_indices, list):
#         for triangle_set, color in zip(triangle_indices, triangle_colors):
#             triangle_set = np.asarray(triangle_set)
#             for i in range(np.asarray(triangle_set).shape[0]):
#                 # import ipdb; ipdb.set_trace()
#                 t_idx = triangle_set[i]
#                 p_idx = triangles[t_idx, :]
#                 vertex_colors[p_idx] = color
#     else:
#         for i in range(triangle_indices.shape[0]):
#             # import ipdb; ipdb.set_trace()
#             t_idx = triangle_indices[i]
#             color = triangle_colors[i, :]
#             p_idx = triangles[t_idx, :]
#             vertex_colors[p_idx] = color
#     if not split_mesh.has_triangle_normals():
#         split_mesh.compute_triangle_normals()
#     split_mesh.compute_vertex_normals()

#     return split_mesh


# def paint_planes(o3d_mesh, planes):
#     # colors = np.arange(0, 0+ len(planes))
#     colors = [0, 3]
#     all_colors = plt.cm.get_cmap('tab10')(colors)[:, :3]

#     # planes_list = [np.copy(plane) for plane in planes]
#     # planes_list = np.

#     new_mesh = assign_some_vertex_colors(o3d_mesh, planes, all_colors)
#     return new_mesh


def down_sample_normals(triangle_normals, down_sample_fraction=0.12, min_samples=10000, flip_normals=False, **kwargs):
    num_normals = triangle_normals.shape[0]
    to_sample = int(down_sample_fraction * num_normals)
    to_sample = max(min([num_normals, min_samples]), to_sample)
    ds_step = int(num_normals / to_sample)
    triangle_normals_ds = np.ascontiguousarray(triangle_normals[:num_normals:ds_step, :])
    if flip_normals:
        triangle_normals_ds = triangle_normals_ds * -1.0
    return triangle_normals_ds


def get_image_peaks(IcoCharts, ga, level=2, with_o3d=False,
                    find_peaks_kwargs=dict(threshold_abs=2, min_distance=1, exclude_border=False, indices=False),
                    cluster_kwargs=dict(t=0.10, criterion='distance'),
                    average_filter=dict(min_total_weight=0.01),
                    **kwargs):

    normalized_bucket_counts_by_vertex = ga.get_normalized_bucket_counts_by_vertex(True)

    t1 = time.perf_counter()
    ico_chart.fill_image(normalized_bucket_counts_by_vertex)  # this takes microseconds
    # plt.imshow(np.asarray(ico_chart.image))
    # plt.show()
    peaks, clusters, avg_peaks, avg_weights = find_peaks_from_ico_charts(IcoCharts, np.asarray(
        normalized_bucket_counts_by_vertex), find_peaks_kwargs=find_peaks_kwargs, cluster_kwargs=cluster_kwargs, average_filter=average_filter)
    t2 = time.perf_counter()

    gaussian_normals_sorted = np.asarray(IcoCharts.sphere_mesh.vertices)
    
    # Create Open3D structures for visualization
    if with_o3d:
        pcd_all_peaks = get_pc_all_peaks(peaks, clusters, gaussian_normals_sorted)
        arrow_avg_peaks = get_arrow_normals(avg_peaks, avg_weights)
    else:
        pcd_all_peaks = None
        arrow_avg_peaks = None

    elapsed_time = (t2 - t1) * 1000
    timings = dict(t_fastga_peak=elapsed_time)

    logging.debug("Peak Detection - Took (ms): %.2f", (t2 - t1) * 1000)

    return avg_peaks, pcd_all_peaks, arrow_avg_peaks, timings


def extract_all_dominant_plane_normals(tri_mesh, level=5, with_o3d=False, ga_=None, ico_chart_=None, **kwargs):

    # Reuse objects if provided
    if ga_ is not None:
        ga = ga_
    else:
        ga = GaussianAccumulatorS2Beta(level=level)

    if ico_chart_ is not None:
        IcoCharts = ico_chart_
    else:
        IcoCharts = IcoCharts(level=level)

    triangle_normals = np.asarray(tri_mesh.triangle_normals)
    triangle_normals_ds = down_sample_normals(triangle_normals, **kwargs)

    # np.savetxt('bad_normals.txt', triangle_normals_ds)
    triangle_normals_ds_mat = MatX3d(triangle_normals_ds)
    t1 = time.perf_counter()
    ga.integrate(triangle_normals_ds_mat)
    t2 = time.perf_counter()

    logging.debug("Gaussian Accumulator - Normals Sampled: %d; Took (ms): %.2f",
                  triangle_normals_ds.shape[0], (t2 - t1) * 1000)


    # New way of detecting peaks, all in C++
    # Only need three parameters now
    fp = kwargs['find_peaks_kwargs']
    cl = kwargs['cluster_kwargs']
    avg_filter = kwargs['average_filter']
    t3 = time.perf_counter()
    avg_peaks = np.array(ga.find_peaks(threshold_abs=fp['threshold_abs'], cluster_distance=cl['t'], min_cluster_weight=avg_filter['min_total_weight']))
    t4 = time.perf_counter()

    # Old, python library (Scipy, sklearn) way of detecting peaks
    # Should still work, this API is not deprecated
    # avg_peaks, pcd_all_peaks, arrow_avg_peaks, timings_dict = get_image_peaks(
    #     ico_chart, ga, level=level, with_o3d=with_o3d, **kwargs)

    # Create Open3D structures for visualization
    if with_o3d:
        # Visualize the Sphere
        accumulator_counts = np.asarray(ga.get_normalized_bucket_counts())
        refined_icosahedron_mesh = create_open_3d_mesh(np.asarray(ga.mesh.triangles), np.asarray(ga.mesh.vertices))
        color_counts = get_colors(accumulator_counts)[:, :3]
        colored_icosahedron = assign_vertex_colors(refined_icosahedron_mesh, color_counts)
    else:
        colored_icosahedron = None

    elapsed_time_fastgac = (t2 - t1) * 1000
    elapsed_time_peak = (t4-t3) * 1000
    elapsed_time_total = elapsed_time_fastgac + elapsed_time_peak

    timings = dict(t_fastgac_total=elapsed_time_total,
                   t_fastgac_integrate=elapsed_time_fastgac, t_fastgac_peak=elapsed_time_peak)

    ga.clear_count()
    # return avg_peaks, pcd_all_peaks, arrow_avg_peaks, colored_icosahedron, timings
    return avg_peaks, None, None, colored_icosahedron, timings


def filter_and_create_polygons(points, polygons, rm=None, line_radius=0.005,
                               postprocess=dict(filter=dict(hole_area=dict(min=0.025, max=100.0), hole_vertices=dict(min=6), plane_area=dict(min=0.05)),
                                                positive_buffer=0.00, negative_buffer=0.00, simplify=0.0)):
    " Apply polygon filtering algorithm, return Open3D Mesh Lines "
    t1 = time.perf_counter()
    planes, obstacles, planes_indices = filter_planes_and_holes(polygons, points, postprocess, rm=rm)
    t2 = time.perf_counter()
    return planes, obstacles, planes_indices, (t2 - t1) * 1000


def extract_planes_and_polygons_from_mesh(tri_mesh, avg_peaks,
                                          polylidar_kwargs=dict(alpha=0.0, lmax=0.1, min_triangles=2000,
                                                                z_thresh=0.1, norm_thresh=0.95, norm_thresh_min=0.95, min_hole_vertices=50, task_threads=4),
                                          filter_polygons=True, pl_=None, optimized=False,
                                          postprocess=dict(filter=dict(hole_area=dict(min=0.025, max=100.0), hole_vertices=dict(min=6), plane_area=dict(min=0.05)),
                                                           positive_buffer=0.00, negative_buffer=0.00, simplify=0.0)):

    if pl_ is not None:
        pl = pl_
    else:
        pl = Polylidar3D(**polylidar_kwargs)

    avg_peaks_mat = MatrixDouble(avg_peaks)
    t0 = time.perf_counter()
    if optimized:
        all_planes, all_polygons = pl.extract_planes_and_polygons_optimized(tri_mesh, avg_peaks_mat)
    else:
        all_planes, all_polygons = pl.extract_planes_and_polygons(tri_mesh, avg_peaks_mat)
    t1 = time.perf_counter()

    # tri_set = pl.extract_tri_set(tri_mesh, avg_peaks_mat)
    # planes_tri_set = [np.argwhere(np.asarray(tri_set) == i)  for i in range(1, 2)]
    # o3d_mesh_painted = paint_planes(o3d_mesh, planes_tri_set)

    polylidar_time = (t1 - t0) * 1000
    # logging.info("Polygon Extraction - Took (ms): %.2f", polylidar_time)
    all_planes_shapely = []
    all_obstacles_shapely = []
    time_filter = []
    time_geometric_planes = []
    # all_poly_lines = []
    geometric_planes = []
    # all_polygons = [[NORMAL_0_POLY_1, NORMAL_0_POLY_2], [NORMAL_1_POLY_1]]
    if filter_polygons:
        vertices = np.asarray(tri_mesh.vertices)
        for i in range(avg_peaks.shape[0]):
            avg_peak = avg_peaks[i, :]
            rm, _ = R.align_vectors([[0, 0, 1]], [avg_peak])  # Rotating matrix the polygon
            polygons_for_normal = all_polygons[i]
            planes = all_planes[i]
            # print(polygons_for_normal)
            if len(polygons_for_normal) > 0:
                planes_shapely, obstacles_shapely, planes_indices, filter_time = filter_and_create_polygons(
                    vertices, polygons_for_normal, rm=rm, postprocess=postprocess)

                t3 = time.perf_counter()
                geometric_planes_for_normal = [extract_geometric_plane(plane_poly[0], planes[plane_idx], tri_mesh, avg_peak) for (
                    plane_poly, plane_idx) in zip(planes_shapely, planes_indices)]
                geometric_planes.append(geometric_planes_for_normal)
                t4 = time.perf_counter()
                time_geometric_planes.append((t4 - t3) * 1000)

                all_planes_shapely.extend(planes_shapely)
                all_obstacles_shapely.extend(obstacles_shapely)
                time_filter.append(filter_time)
                # all_poly_lines.extend(poly_lines)

    timings = dict(t_polylidar_planepoly=polylidar_time, t_polylidar_filter=np.array(time_filter).mean(), t_geometric_planes=np.array(time_geometric_planes).sum())
    # all_planes_shapely, all_obstacles_shapely, all_poly_lines, timings
    return all_planes_shapely, all_obstacles_shapely, geometric_planes, timings