import os
import cv2


from mediapipe_facemesh.face_mesh_restructure import FaceMeshRestructure
from utils.save_mesh import save_ply_mesh
from utils.visualize import get_depth_map


if __name__ == '__main__':
    # Example usage:
    files = os.listdir('data/sample') 

    for idx, file in enumerate(files):
        if file.split('.')[-1] not in ['jpg', 'png']:
            continue

        path_read = f'data/sample/{file}'
        image = cv2.imread(path_read)
        write_image = image.copy()

        detector = FaceMeshRestructure(max_loop = 2)
        input_image = image.copy()
        output_image, face_detected, points, depth_list, triangles = detector.generate_face_mesh(input_image, False, False)
        cv2.imwrite(f'data/{file}_face_dense.jpg', output_image)

        depth_map = get_depth_map(input_image, points, triangles.simplices, depth_list)
        cv2.imwrite(f'data/{file}_face_depth.jpg', depth_map)

        points_3d = []
        for i, (x,y) in enumerate(points):
            z = 450 / 2 - depth_list[i] * 450
            points_3d.append([x, y, z])
        save_ply_mesh(points_3d,  triangles.simplices, f'data/{file}_face_dense.ply')
