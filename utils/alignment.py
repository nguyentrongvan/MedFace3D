import numpy as np
import transforms3d as t3d
from utils.transform import get_matrix_rotate_point_around_x, \
                            get_matrix_rotate_point_around_y, \
                            get_matrix_rotate_point_around_z

from scipy.spatial.transform import Rotation as R
from utils.render_mesh import rotate_point_cloud

FRONTAL_FACE = {
    "x" : 0,
    "y" : 1,
    "z" : 0
}


def get_frontal_landmarks(landmarks_initial_pose, face_pose):
    # Calculate the centroid (average) of all 3D landmarks
    desired_x_degrees = face_pose[0] - face_pose[0] # FRONTAL_FACE['x'] - face_pose[0]
    desired_y_degrees = FRONTAL_FACE['y'] - face_pose[1]
    desired_z_degrees = face_pose[2] - face_pose[2]

    # Convert degrees to radians
    rotation_matrix_x = get_matrix_rotate_point_around_x(desired_x_degrees)
    rotation_matrix_y = get_matrix_rotate_point_around_y(desired_y_degrees)
    rotation_matrix_z = get_matrix_rotate_point_around_z(desired_z_degrees)
    
    landmarks_final_pose = rotate_point_cloud(landmarks_initial_pose,   rotation_matrix_x)
    landmarks_final_pose = rotate_point_cloud(landmarks_final_pose,     rotation_matrix_y)
    landmarks_final_pose = rotate_point_cloud(landmarks_final_pose,     rotation_matrix_z)

    return  landmarks_final_pose


def get_frontal_landmarks_v2(landmarks_initial_pose, face_pose):
    current_angles = face_pose

    Rz = R.from_euler('z', -current_angles[2], degrees=False)
    Ry = R.from_euler('y', -current_angles[1], degrees=False)
    Rx = R.from_euler('x', -current_angles[0], degrees=False)

    # Combine the rotations in reverse order (roll, pitch, yaw)
    result_rotation = Rz * Ry * Rx
    result_euler_angles = result_rotation.as_euler('zyx', degrees=False)

    desired_x_degrees = result_euler_angles[0]
    desired_y_degrees = result_euler_angles[1]
    desired_z_degrees = result_euler_angles[2]

    # Convert degrees to radians
    rotation_matrix_x = get_matrix_rotate_point_around_x(desired_x_degrees)
    rotation_matrix_y = get_matrix_rotate_point_around_y(desired_y_degrees)
    rotation_matrix_z = get_matrix_rotate_point_around_z(desired_z_degrees)
    
    landmarks_final_pose = rotate_point_cloud(landmarks_initial_pose,   rotation_matrix_x)
    landmarks_final_pose = rotate_point_cloud(landmarks_final_pose,     rotation_matrix_y)
    landmarks_final_pose = rotate_point_cloud(landmarks_final_pose,     rotation_matrix_z)

    return  landmarks_final_pose


# refer: https://github.com/yfeng95/PRNet.git
def frontalize(vertices, frontal_vertices):
    vertices_homo = np.hstack((vertices, np.ones([vertices.shape[0],1]))) #n x 4
    P = np.linalg.lstsq(vertices_homo, frontal_vertices)[0].T # Affine matrix. 3 x 4
    front_vertices = vertices_homo.dot(P.T)

    return front_vertices

