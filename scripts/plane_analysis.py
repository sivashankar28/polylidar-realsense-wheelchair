import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from joblib import load
from mpl_toolkits.mplot3d import Axes3D

def make_square(cent, ax1, ax2, normal, w=1, h=1):
    p1 = cent + h * ax1 + 0.5 * w * ax2 
    p2 = cent + h * ax1 - 0.5 * w * ax2
    p3 = cent + 0.5 * w * ax2 
    p4 = cent - 0.5 * w * ax2
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    vec1 = vectors_proj[0, :]
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vectors_proj[1, :] 
    vec2 = vec2 / np.linalg.norm(vec2)

    ax.quiver(0, 0, 0, ground_normal[0], ground_normal[1], ground_normal[2], length=0.5, color='orange')
    ax.quiver(0, 0, 0, vectors[0, 0], vectors[0, 1], vectors[0,2], length=0.5, color='blue')
    ax.quiver(0, 0, 0, vectors[1, 0], vectors[1, 1], vectors[1,2], length=0.5, color='red')

    ax.quiver(0, 0, 0, vec1[0], vec1[1], vec1[2], length=1, color='blue')
    ax.quiver(0, 0, 0, vec2[0], vec2[1], vec2[2], length=1, color='red')
    fig.show()

    a = np.dot(vec1, vec2)
    theta = np.degrees(np.arccos(a))
    return dist, theta

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    data = load('data/planes.joblib')
    first_points = data['first_plane']['all_points']
    second_points = data['second_plane']['all_points']

    normal = -data['first_plane']['normal_ransac']
    first_points_mean = np.mean(second_points, axis=0)
  
    first_points_ = first_points + np.ones_like(first_points) * normal
    first_points = np.concatenate([first_points, first_points_], axis=0)

    second_points_ = second_points + np.ones_like(second_points) * normal
    second_points = np.concatenate([second_points, second_points_], axis=0)

    X = np.concatenate([first_points, second_points],axis=0) # will be a (N+K) X 3 numpy array
    first_points_y = np.zeros((first_points.shape[0],), dtype=int)
    second_points_y = np.ones((second_points.shape[0],), dtype=int)
    Y = np.concatenate([first_points_y, second_points_y], axis=0)

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)

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
    print(normal_svm)
    print(-d)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    
    xyz = normal_svm * offset
    cent = project_points_geometric_plane(np.expand_dims(first_points_mean, axis=0), normal_svm, xyz)[0, :]
    
    cross = np.cross(normal_svm, normal)
    square_points = make_square(cent, normal, cross, normal_svm) 

    dist, theta = get_theta_and_distance(normal_svm, cent, normal)
    print(dist, theta)
    ax.quiver(xyz[0], xyz[1], xyz[2], normal_svm[0], normal_svm[1], normal_svm[2], length=1)
    ax.quiver(cent[0], cent[1], cent[2], normal_svm[0], normal_svm[1], normal_svm[2], length=1)
    ax.quiver(cent[0], cent[1], cent[2], normal[0], normal[1], normal[2], length=1)
    ax.quiver(cent[0], cent[1], cent[2], cross[0], cross[1], cross[2], length=1)
    ax.scatter(square_points[:,0], square_points[:,1], square_points[:,2], c='k', s=100)

    ax.scatter(0,0,0, c=0)

    xx = np.arange(-2, 2, 0.25)
    yy = np.arange(-2, 2, 0.25)
    xx, yy = np.meshgrid(xx, yy)

    # a *x + b*y + c*z +d = 0
    # z = (-d -a*x -b*y)/c
    zz = (-d - a*xx -b *yy) / c

    ax.plot_surface(xx, yy, zz, alpha=0.2)
    stuff = np.dot(normal,normal_svm)
    print(stuff, "Should be close to 0")
    
    plt.show()

if __name__ == "__main__":
    main()


