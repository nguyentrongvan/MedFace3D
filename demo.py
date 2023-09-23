import os
import cv2
import argparse

from mediapipe_facemesh.face_mesh_restructure import FaceMeshRestructure
from utils.visualize import save_images_as_gif, get_depth_map
from utils.render_mesh import render_mesh, render_rotate_mesh
from utils.transform import get_3D_point_cloud

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',  '--folder',      default='data/sample' , help='path to sample folder image')
    parser.add_argument('-l',  '--max_loop',    type=int, default= 2,   help='max loop to reconstruct 3D face mesh')
    parser.add_argument('-p',  '--point_cloud', action="store_true",  default=True,  help='get image result as point cloud')
    parser.add_argument('-d',  '--depth_scale', action="store_true",  default=False,  help='depth scale for depth value')


    args = parser.parse_args()

    files = os.listdir(args.folder) 

    for _, file in enumerate(files):
        if file.split('.')[-1] not in ['jpg', 'png']:
            continue

        path_read = f'{args.folder}/{file}'
        image = cv2.imread(path_read)
        _, w, _ = image.shape

        detector = FaceMeshRestructure(max_loop = args.max_loop)
        input_image = image.copy()
        output_image, face_detected, points, depth_list, triangles = detector.generate_face_mesh(input_image, args.point_cloud, args.depth_scale)
        
        if face_detected:
            list_view = []
            list_view.append(image)
            list_view.append(output_image)
            
            depth_map = get_depth_map(input_image, points, triangles.simplices, depth_list)
            list_view.append(depth_map)

            point_cloud_data = get_3D_point_cloud(points, depth_list, w)
            mesh_image = render_mesh(image, point_cloud_data)
            list_view.append(mesh_image)
       
            view_face = render_rotate_mesh(image, point_cloud_data)
            list_view.extend(view_face)

            save_images_as_gif(list_view, f'{args.folder}/{file}_face_reconstruction.gif')

        else: print(f'No face detected in is image')
if __name__ == '__main__':
    # Example usage:
    main()
