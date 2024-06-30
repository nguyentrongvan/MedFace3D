import numpy as np

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

        return yaw, pitch, roll



