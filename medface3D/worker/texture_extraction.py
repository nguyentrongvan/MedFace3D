import numpy as np
import cv2

class TextureExtractor:
    def __init__(self) -> None:
        pass

    def extract_boundary(self, point_cloud):
        projected_points = point_cloud[:, :2]  # Keep only the X and Y coordinates

        # Sort the points in a clockwise direction
        center = np.mean(projected_points, axis=0)
        sorted_points = sorted(projected_points, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
        sorted_points = np.array(sorted_points, dtype=np.float32)

        hull = cv2.convexHull(sorted_points, clockwise=True)
        # Create a boundary list
        boundary = []
        
        if len(hull) > 0:
            for point_bd in hull:
                bd_x, bd_y = point_bd[0].astype(int)
                boundary.append([bd_x, bd_y])
        return np.asarray(boundary)
    

    def extract_face_area(self, image, boundary_points):
        height, width, _ = image.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        cv2.fillPoly(mask, [boundary_points], 255)  
        face_area = cv2.bitwise_and(image, image, mask=mask)
        return face_area


    def get_projected_points(self, point_cloud, fx=10.0, fy=10.0, cx=450, cy=450):
        projected_points = []  # List to store 2D pixel coordinates

        for point in point_cloud:
            x, y, z = point
            u = int(fx * x / z + cx)
            v = int(fy * y / z + cy)
            projected_points.append((u, v))
        return projected_points
    

    def calculate_u_coordinate(self, point, x_min, x_max):
        # Assuming the X coordinate of the point is in the range [x_min, x_max]
        x = point[0]

        u = (x - x_min) / (x_max - x_min)
        u = max(0.0, min(1.0, u))
        return u


    def calculate_v_coordinate(self, point, y_min, y_max):
        # Assuming the Y coordinate of the point is in the range [y_min, y_max]
        y = point[1]

        v = (y - y_min) / (y_max - y_min)
        v = max(0.0, min(1.0, v))
        return v
    

    def extract_texture(self, original_image, point_cloud, square_size = 256):
        uv_coordinates = []  # List to store 2D UV texture coordinates
        original_image_height, original_image_width, _ = original_image.shape

        x_min = min(point[0] for point in point_cloud)
        x_max = max(point[0] for point in point_cloud)
        y_min = min(point[1] for point in point_cloud)
        y_max = max(point[1] for point in point_cloud)

        for point in point_cloud:
            # Calculate UV coordinates based on the properties of your 3D model
            u = self.calculate_u_coordinate(point, x_min, x_max)
            v = self.calculate_v_coordinate(point, y_min, y_max)
            uv_coordinates.append((u, v))

        texture_image = np.zeros((original_image_height, original_image_width, 3), dtype=np.uint8)

        for u, v in uv_coordinates:
            x = int(u * (original_image_width - 1))  # Map u to image width
            y = int(v * (original_image_height - 1))  # Map v to image height

            if 0 <= x < original_image_width and 0 <= y < original_image_height:
                texture_image[y, x] = original_image[y, x]

        texture_image = cv2.resize(texture_image, (square_size, square_size))
        return texture_image












