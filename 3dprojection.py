import numpy as np
import matplotlib.pyplot as plt
from enum import Enum,auto

class plane_2d(Enum):
    XY = auto()
    XZ = auto()
    YZ = auto()

def isect_line_plane_v3(rayPoint, rayDirection, planeNormal, planePoint, epsilon=1e-6):

    ndotu = planeNormal@rayDirection

    if abs(ndotu) < epsilon:
        print("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal@w / ndotu

    Psi = w + si * rayDirection + planePoint
    return Psi


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 /
                                                      np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def plane_intersection(target_point, center_point, aim_points,fov_vector, plane=plane_2d.XY):

    target_vector = center_point - target_point

    intersect_points = []
    for helper in aim_points:
        intersect = isect_line_plane_v3(
            helper, helper-target_point, target_vector, center_point)

        intersect_points.append(intersect)
    intersect_vectors = [(vec - center_point) for vec in intersect_points]

    if plane == plane_2d.XY:
        normal_vec_2d = np.array([[0, 0, 1]])
    elif plane == plane_2d.XZ:
        normal_vec_2d = np.array([[0, 1, 0]])

    mat = rotation_matrix_from_vectors(fov_vector, normal_vec_2d)
    flat = [mat@vec for vec in intersect_vectors]

    fig1 = plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.scatter(target_point[0], target_point[1], target_point[2], label = "target")
    ax.scatter(center_point[0], center_point[1], center_point[2], label = "center")
    for helper in aim_points:
        ax.scatter(helper[0], helper[1], helper[2], c="black", label = "aim")

    for helper in intersect_points:
        ax.scatter(helper[0], helper[1], helper[2], c="red", label = "isect")

    plt.xlabel("X")
    plt.ylabel("Y")
    ax.legend()
    fig1.show()


    fig2 = plt.figure(2)
    x = [float(i[0]) for i in flat]
    y = [float(i[1]) for i in flat]
    plt.scatter(x,y,c="blue")
    
    plt.scatter(0,0,c="red")
    fig2.show()
    input()

if __name__ == "__main__":

    center_point = np.array([1, 1, 1])
    target_point = np.array([4, 4, 7])
    fov_vector = np.array([2,3,6])
    helper_list = [np.array([3+2*i, 4-i, 5+i]) for i in range(5)]


    plane_intersection(center_point, target_point, helper_list, fov_vector)