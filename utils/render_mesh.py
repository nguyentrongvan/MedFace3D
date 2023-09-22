import numpy as np
import cv2
import scipy.spatial


def render_mesh(image_ori, point_cloud_data, blank_background = True):
    image_size = (image_ori.shape[1], image_ori.shape[0])
    
    if not blank_background:
        image = image_ori.copy()
    else: image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    # Define a simple light direction (you can adjust this as needed)
    light_direction = np.array([0, 0, 1])
    camera_position = np.array([image_size[0] / 2, image_size[1] / 2, 0])

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

        # Draw the triangle with shading
        cv2.fillPoly(image, [vertices[:, :2].astype(int)], color=(intensity * 200, intensity * 255, intensity * 50))
    return image
