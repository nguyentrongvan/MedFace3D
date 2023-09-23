import numpy as np

def min_max_scale(value_list):
    max_value = max(value_list)
    min_value = min(value_list)
    scaled_list = [(x - min_value) / (max_value - min_value) for x in value_list]
    return np.asarray(scaled_list)

def get_matrix_rotate_point_around_x(angle_radians):
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_radians), -np.sin(angle_radians)],
        [0, np.sin(angle_radians), np.cos(angle_radians)]
    ])
    return rotation_matrix

def get_matrix_rotate_point_around_y(angle_radians):
    rotation_matrix = np.array([
        [np.cos(angle_radians), 0, np.sin(angle_radians)],
        [0, 1, 0],
        [-np.sin(angle_radians), 0, np.cos(angle_radians)]
    ])
    return rotation_matrix


def get_matrix_rotate_point_around_z(angle_radians):
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])
    return rotation_matrix


def get_3D_point_cloud(points, depth_list, const = 450):
    points_3d = []
    for i, (x,y) in enumerate(points):
        z = const / 2 - depth_list[i] * const
        points_3d.append([x, y, z])
    
    return np.asarray(points_3d)