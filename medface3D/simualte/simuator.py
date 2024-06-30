import numpy as np
from scipy.spatial.transform import Rotation

def merge_action(point1, point2):
    # Compute the centroid of each point cloud
    centroid1 = np.mean(point1, axis=0)
    centroid2 = np.mean(point2, axis=0)

    # Compute the translation vector between the centroids
    translation_vector = centroid1 - centroid2

    # Calculate the rotation matrix between the two point clouds
    # Here we assume that the point clouds have the same number of points and are aligned
    # You can use more advanced techniques (e.g., ICP) for better alignment if needed
    H = np.dot(point1.T, point2)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Create a rotation object using scipy's Rotation class
    rotation = Rotation.from_matrix(R)

    # Apply the translation and rotation to each point in point2 individually
    point2_transformed = []
    for point in point2:
        transformed_point = rotation.apply(point - centroid2) + centroid1
        point2_transformed.append(transformed_point)

    # Convert the result back to a NumPy array
    point2_transformed = np.array(point2_transformed)
    return point2_transformed