import numpy as np
import cv2
import scipy.spatial
from utils.transform import get_matrix_rotate_point_around_x, \
                            get_matrix_rotate_point_around_y, \
                            get_matrix_rotate_point_around_z

def render_mesh(image_ori, point_cloud_data, blank_background = True):
    image_size = (image_ori.shape[1], image_ori.shape[0])
    
    if not blank_background:
        image = image_ori.copy()
    else: image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    # Define a simple light direction (you can adjust this as needed)
    light_direction = np.array([0, 0, 1])
    # Create a Delaunay triangulation to connect the points and form a mesh
    triangles = scipy.spatial.Delaunay(point_cloud_data[:, :2]) 
    
    for simplex in triangles.simplices:
        vertices = point_cloud_data[simplex]

        # Compute the face normal (in this case, we use the cross product of two edges)
        edge1 = vertices[1] - vertices[0]
        edge2 = vertices[2] - vertices[0]

        face_normal = np.cross(edge1, edge2)
        face_normal /= np.linalg.norm(face_normal)

        # Calculate the intensity based on the dot product with the light direction
        intensity = np.dot(face_normal, light_direction)

        # Clip the intensity to avoid negative values
        intensity = max(intensity, 0)
        color=(intensity * 200, intensity * 255, intensity * 50)

        # Draw the triangle with shading
        cv2.fillPoly(image, [vertices[:, :2].astype(int)], color = color)
    return image


def calculate_center(point_cloud):
    # Calculate the center by finding the average of all point coordinates
    num_points = len(point_cloud)
    center = np.mean(point_cloud, axis=0)
    return center


def rotate_point_cloud(point_cloud, rotation_matrix):
    # Translate point cloud to the origin (center)
    center = calculate_center(point_cloud)
    translated_point_cloud = point_cloud - center

    # Apply rotation to all points
    rotated_point_cloud = np.dot(translated_point_cloud, rotation_matrix)

    # Translate points back to their original positions
    rotated_point_cloud += center

    return rotated_point_cloud


def render_rotate_mesh(image_ori, point_cloud_data, delta_angle = 10, axis = 'y'):
    list_view = []
    for angle_degrees in range(-90, 90, delta_angle):
        angle_radians = np.radians(angle_degrees)
        
        if  axis == 'x':
            rotation_matrix = get_matrix_rotate_point_around_x(angle_radians)
        elif axis == 'y':
            rotation_matrix = get_matrix_rotate_point_around_y(angle_radians)
        elif axis == 'z':
            rotation_matrix = get_matrix_rotate_point_around_z(angle_radians)
        else: raise Exception(f'axis is invalid, please use `x`, `y` or `z`')

        # Rotate the point cloud
        rotated_point_cloud = rotate_point_cloud(point_cloud_data, rotation_matrix)
        new_mesh = render_mesh(image_ori, rotated_point_cloud, True)    
        list_view.append(new_mesh)
    return list_view



        
    

    

