import numpy as np
import math

class FacePoseEstimator:
    def __init__(self) -> None:
        pass

    def get_pose(self, point_cloud):
        centroid = np.mean(point_cloud, axis=0)
        covariance_matrix = np.cov(point_cloud - centroid, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        roll = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        pitch = np.degrees(np.arctan2(-eigenvectors[2, 0], np.sqrt(eigenvectors[2, 1] ** 2 + eigenvectors[2, 2] ** 2)))
        yaw = np.degrees(np.arctan2(eigenvectors[2, 1], eigenvectors[2, 2]))

        return roll, pitch, yaw


def get_face_pose(face_landmarks):
    # Extract relevant facial landmarks
    landmarks = face_landmarks.landmark
    left_eye = landmarks[33]  # Left eye landmark
    right_eye = landmarks[263]  # Right eye landmark
    nose_tip = landmarks[2]  # Nose tip landmark

    # Calculate yaw angle (left-right head rotation)
    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y
    yaw_angle = math.degrees(math.atan2(dy, dx))

    # Calculate pitch angle (up-down head rotation)
    dz = right_eye.z - left_eye.z
    dy = right_eye.y - left_eye.y
    pitch_angle = math.degrees(math.atan2(dz, dy))

    # Calculate roll angle (tilt of the head)
    dx = right_eye.x - left_eye.x
    dz = right_eye.z - left_eye.z
    roll_angle = math.degrees(math.atan2(dz, dx))
    return yaw_angle, pitch_angle, roll_angle


