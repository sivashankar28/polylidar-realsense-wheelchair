import time
import logging
import itertools
import sys
from shapely.geometry import Polygon
import numpy as np
from polylidar import HalfEdgeTriangulation

from surfacedetector.utility.helper_ransac import estimate_plane

from sklearn import svm
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axis3d

def extract_geometric_plane(polygon: Polygon, plane_triangle_indices, tri_mesh: HalfEdgeTriangulation, normal: np.ndarray):
    """Will extract geometric details from the polygon and plane of interest
    Args:
        polygon (Polygon): Shapely Polygon of a flat surface
        plane_triangle_indices (ndarray uint64): Triangle indices of the plane in the mesh
        tri_mesh (HalfEdgeTriangulation): The mesh of the environment
        normal (np.ndarray): The surface normal that this plane was extracted on
    Returns:
        [type]: [description]
    """         
    # triangles:np.ndarray = np.asarray(tri_mesh.triangles)
    # vertices:np.ndarray = np.asarray(tri_mesh.vertices)
    # all_point_indices = triangles[plane_triangle_indices, :]
    # all_point_indices = np.reshape(all_point_indices, (np.prod(all_point_indices.shape), ))
    # all_point_indices = np.unique(all_point_indices)
    # all_points = vertices[all_point_indices, :]

    all_points = np.asarray(polygon.exterior.coords)
    # centroid = np.mean(all_points, axis=0) # TODO polygon.centroid ?
    normal_ransac, centroid, _ = estimate_plane(all_points)

    return dict(point=centroid, normal=normal, all_points=all_points, area=polygon.area, normal_ransac=normal_ransac)


def analyze_planes(geometric_planes):
    """This will analyze all geometric planes that have been extracted and find the curb height
    Args:
        geometric_planes (List[dict]): A list of dicts representing the geometric planes
    Returns:
        float: Height of curb in meters
    """

    # This code will find the ground normal index, the index into geometric_planes
    # with the largest area of surfaces (e.g., the street and sidewalk)
    # if len(geometric_planes) < 2:
    #     return 0.0, None, None
    max_area = 0.0
    ground_normal_index = 0
    mean_normal_ransac = np.array([0.0, 0.0, 0.0])
    at_least_two_planes = False
    for i, geometric_planes_for_normal in enumerate(geometric_planes):
        if len(geometric_planes_for_normal) > 1:
            at_least_two_planes = True
            total_area = 0.0
            total_normal_ransac = np.array([0.0, 0.0, 0.0])
            for j, plane in enumerate(geometric_planes_for_normal):
                logging.debug(
                    f"Plane {j} - Normal: {plane['normal']:}, Ransac Normal: {plane['normal_ransac']:}, Point: {plane['point']:}")
                # np.save(f'scratch/plane_{i}_{j}.npy', plane['all_points'])
                total_normal_ransac += plane['normal_ransac'] * plane['area']
                total_area += plane['area']
            if total_area > max_area:
                max_area = total_area
                ground_normal_index = i
                mean_normal_ransac = total_normal_ransac / total_area
                mean_normal_ransac = mean_normal_ransac / np.linalg.norm(mean_normal_ransac) * -1
    if not at_least_two_planes:
        return 0.0, None, None
    # This code will find the maximum orthogonal distance between any tow pair of surfaces with
    # the same normal
    # import ipdb; ipdb.set_trace()
    max_orthogonal_distance = 0.0
    geometric_planes_for_normal = geometric_planes[ground_normal_index]
    first_plane_final = None
    second_plane_final = None
    first_plane_final_area = 0.0
    second_plane_final_area = 0.0
    
    for pair in itertools.combinations(range(len(geometric_planes_for_normal)), 2):
        # print(pair)
        first_plane = geometric_planes_for_normal[pair[0]]
        second_plane = geometric_planes_for_normal[pair[1]]
        orthoganal_distance = np.abs(mean_normal_ransac.dot(first_plane['point'] - second_plane['point']))

        if orthoganal_distance > max_orthogonal_distance:
            max_orthogonal_distance = orthoganal_distance
            first_plane_final = first_plane
            second_plane_final = second_plane
    

    logging.debug(f"Curb Height: {max_orthogonal_distance:}")
    return max_orthogonal_distance, first_plane_final, second_plane_final

def make_square(cent, ax1, ax2, normal, w=1.0, h=0.25):
    p1 = cent + h * ax1 + 0.5 * w * ax2 
    p2 = cent + h * ax1 - 0.5 * w * ax2
    p3 = cent - 0.5 * w * ax2 
    p4 = cent + 0.5 * w * ax2
    points = np.array([p1,p2,p3,p4])
    projected_points = project_points_geometric_plane(points, normal, cent)
    return projected_points

def project_points_geometric_plane(points, normal, point_on_plane):
    diff = points - point_on_plane
    dist = np.dot(diff, normal)
    scaled_vector = normal*dist[:,np.newaxis]
    # import ipdb; ipdb.set_trace()
    projected_points = points - scaled_vector
    
    return projected_points

def get_theta_and_distance(plane_normal, point_on_plane, ground_normal):
    diff = np.array([0.0,0.0,0.0]) - point_on_plane
    dist = np.dot(diff, plane_normal)
    dist = np.abs(dist)

    vectors = np.array([[0.0, 0.0, -1.0], plane_normal])
    vectors_proj = project_points_geometric_plane(vectors, ground_normal, np.array([0.0, 0.0, 0.0]))

    vec1 = vectors_proj[0, :]
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vectors_proj[1, :] 
    vec2 = vec2 / np.linalg.norm(vec2)

    a = np.dot(vec1, vec2)
    # import ipdb; ipdb.set_trace()
    theta = np.degrees(np.arccos(a))
    cross = np.cross(vec1, vec2)

    if (np.dot(ground_normal, cross) < 0):
        theta = -theta

    return dist, theta

        
def hplane(first_plane, second_plane):
    if first_plane is None or second_plane is None:
        return

    first_points = first_plane['all_points']   # NX3 numpy array
    second_points = second_plane['all_points'] # KX3 numpy array
    first_points_mean = np.mean(second_points, axis=0)
    normal = -first_plane['normal_ransac']
    

    first_points_ = first_points + np.ones_like(first_points) * normal
    first_points = np.concatenate([first_points, first_points_], axis=0)

    second_points_ = second_points + np.ones_like(second_points) * normal
    second_points = np.concatenate([second_points, second_points_], axis=0)

    # Make a duplicate plane for the street level and sidewalk level so we can fit a hyperplane between it
    X = np.concatenate([first_points, second_points],axis=0) # will be a (N+K) X 3 numpy array
    first_points_y = np.zeros((first_points.shape[0],), dtype=int)
    second_points_y = np.ones((second_points.shape[0],), dtype=int)
    Y = np.concatenate([first_points_y, second_points_y], axis=0)

    C = 1.0 #SVM regularization parameter
    clf= svm.SVC(kernel='linear', C=C).fit(X,Y)
     
    #Fit the model
    clf.fit(X, Y)

    #Get the seperating plane
    a = clf.coef_[0][0]
    b = clf.coef_[0][1]
    c = clf.coef_[0][2]
    d = clf.intercept_[0]
    
    normal_svm = np.array([a,b,c])
    length_normal = np.linalg.norm(normal_svm)
    normal_svm = normal_svm / np.linalg.norm(normal_svm)
    offset = -d / length_normal

    xyz = normal_svm * offset

    center = project_points_geometric_plane(np.expand_dims(first_points_mean, axis=0), normal_svm, xyz)[0, :]
    
    cross = np.cross(normal_svm, normal)
    square_points = make_square(center, normal, cross, normal_svm)    
    
    return square_points, normal_svm, center