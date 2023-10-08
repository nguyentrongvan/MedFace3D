import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial.distance import cdist


def smoothing_point_cloud(point_cloud_data):
    # Convert the point cloud data to a NumPy array for easier manipulation
    point_cloud_array = np.array(point_cloud_data)

    # Define the grid resolution for interpolation (adjust as needed)
    grid_resolution = 0.1  # Adjust as needed

    # Create a regular grid of points within the range of the point cloud
    x_range = np.arange(min(point_cloud_array[:, 0]), max(point_cloud_array[:, 0]), grid_resolution)
    y_range = np.arange(min(point_cloud_array[:, 1]), max(point_cloud_array[:, 1]), grid_resolution)

    # Generate interpolation points as a list of (x, y) coordinates
    interpolation_points = np.array([[x, y] for x in x_range for y in y_range])

    # Extract the depth values for interpolation
    depth_values = point_cloud_array[:, 2]

    # Perform depth value interpolation using griddata
    interpolated_depth = griddata(point_cloud_array[:, :2], depth_values, interpolation_points, method='nearest') 
    interpolated_point_cloud_data = np.column_stack((interpolation_points, interpolated_depth))
    
    return interpolated_point_cloud_data


