import numpy as np
import transforms3d as t3d
from utils.transform import get_matrix_rotate_point_around_x, \
                            get_matrix_rotate_point_around_y, \
                            get_matrix_rotate_point_around_z

from utils.render_mesh import rotate_point_cloud

# Function to calculate the rotation matrix for yaw, pitch, and roll
def euler_to_rotation_matrix(yaw, pitch, roll):
    # Calculate the rotation matrix for yaw, pitch, and roll
    rotation_matrix = np.array([
        [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)],
        [np.sin(roll) * np.sin(pitch) * np.cos(yaw) - np.cos(roll) * np.sin(yaw), 
         np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll) * np.cos(yaw), 
         np.sin(roll) * np.cos(pitch)],
        [np.cos(roll) * np.sin(pitch) * np.cos(yaw) + np.sin(roll) * np.sin(yaw), 
         np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.sin(roll) * np.cos(yaw), 
         np.cos(roll) * np.cos(pitch)]
    ])
    return rotation_matrix


def get_fontal_landmarks(landmarks_initial_pose, face_pose):
    # Calculate the centroid (average) of all 3D landmarks
    desired_yaw_degrees     = 1.6362334793124806    - face_pose[0]
    desired_pitch_degrees   = 55.34098251173929     - face_pose[1]
    desired_roll_degrees    = 2.365934232681887     - face_pose[2]
    # Convert degrees to radians
    rotation_matrix_x = get_matrix_rotate_point_around_x(desired_roll_degrees)
    rotation_matrix_y = get_matrix_rotate_point_around_y(desired_yaw_degrees)
    rotation_matrix_z = get_matrix_rotate_point_around_z(desired_pitch_degrees)
    
    landmarks_final_pose = rotate_point_cloud(landmarks_initial_pose,   rotation_matrix_x)
    landmarks_final_pose = rotate_point_cloud(landmarks_final_pose,     rotation_matrix_y)
    landmarks_final_pose = rotate_point_cloud(landmarks_final_pose,     rotation_matrix_z)

    return  landmarks_final_pose
