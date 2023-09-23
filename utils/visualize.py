import cv2
import numpy as np
from PIL import Image

from utils.transform import min_max_scale

def plot_mesh(image, points, triangle, depth_list):
    for idx, simplex in enumerate(triangle):
        pt1, pt2, pt3 = points[simplex]
        depth1, depth2, depth3 = depth_list[simplex]

        # Map depth to a color
        depth_color_1 = int((depth1 + 1) * 127.5)  # Assuming depths range from -1 to 1
        depth_color_1 = (depth_color_1, 0, 0)

        depth_color_2 = int((depth2 + 1) * 127.5)  # Assuming depths range from -1 to 1
        depth_color_2 = (depth_color_2, 0, 0)

        depth_color_3 = int((depth3 + 1) * 127.5)  # Assuming depths range from -1 to 1
        depth_color_3 = (depth_color_3, 0, 0)

        cv2.line(image, tuple(pt1), tuple(pt2), depth_color_1, 1)
        cv2.line(image, tuple(pt2), tuple(pt3), depth_color_2, 1)
        cv2.line(image, tuple(pt3), tuple(pt1), depth_color_3, 1)
    return image


def plot_point_cloud(image, points, depth_list):
    for idx, point in enumerate(points):
        x, y = point
        z = 255 - depth_list[idx] * 255
        depth_color = (z, 0, 0)
        cv2.circle(image, (x,y), 1, depth_color, -1)
    return image


def get_depth_color(depth_value):
    z = 255 - depth_value * 255
    depth_color = (z, 255, 255)
    return depth_color


def get_depth_map(image, points, triangle, depth_list):
    background = np.zeros((image.shape[0], image.shape[1], 1))
    depth_list = min_max_scale(depth_list)

    for _, simplex in enumerate(triangle):
        pt1, pt2, pt3 = points[simplex]
        depth1, depth2, depth3 = depth_list[simplex]

        depth_color_1 = get_depth_color(depth1)
        depth_color_2 = get_depth_color(depth2)
        depth_color_3 = get_depth_color(depth3)

        cv2.line(background, tuple(pt1), tuple(pt2), depth_color_1, 3)
        cv2.line(background, tuple(pt2), tuple(pt3), depth_color_2, 3)
        cv2.line(background, tuple(pt3), tuple(pt1), depth_color_3, 3)
    return np.concatenate([background] * 3, axis=-1)


def save_images_as_gif(image_list, output_filename, duration=400):
    # Convert NumPy arrays to PIL Images
    pil_images = [Image.fromarray(np.uint8(image[:, :, ::-1])) for image in image_list]

    pil_images[0].save(
        output_filename,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0
    )