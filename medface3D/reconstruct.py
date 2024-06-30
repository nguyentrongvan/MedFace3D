import os
import cv2
import argparse
import numpy as np

from medface3d.worker.face_mesh_reconstruction import FaceMeshGenerator
from medface3d.utils.save_mesh import save_ply_mesh
from medface3d.utils.visualize import get_depth_map, save_images_as_gif
from medface3d.utils.render_mesh import render_mesh, render_rotate_mesh
from medface3d.utils.transform import get_3D_point_cloud

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',  '--folder',      default='data/sample' , help='path to sample folder image')
    parser.add_argument('-l',  '--max_loop',    type=int, default= 2,   help='max loop to reconstruct 3D face mesh')
    parser.add_argument('-p',  '--point_cloud', action="store_true",    help='get image result as point cloud')
    parser.add_argument('-d',  '--depth_scale', action="store_true",    help='depth scale for depth value')
    parser.add_argument('-sv', '--save_mesh',   action="store_true",    help='save result as .ply file')
    parser.add_argument('-dm', '--depth_map',   action="store_true",    help='save result as depth map')
    parser.add_argument('-sf', '--face_dense',  action="store_true",    help='save result as dense map')
    parser.add_argument('-rm', '--render_mesh',  action="store_true",    help='render 3D mesh map', default=True)
    parser.add_argument('-mv', '--mesh_view',  action="store_true",    help='render 3D mesh map', default=True)
    parser.add_argument('-ar', '--axis_rotate',  default='y',    help='render 3D mesh map')


    args = parser.parse_args()

    files = os.listdir(args.folder)

    for _, file in enumerate(files):
        if file.split('.')[-1] not in ['jpg', 'png']:
            continue

        path_read = f'{args.folder}/{file}'
        image = cv2.imread(path_read)
        h, w, c = image.shape

        detector = FaceMeshGenerator(max_loop = args.max_loop)
        input_image = image.copy()
        output_image, face_detected, points, depth_list, triangles = detector.generate_face_mesh(input_image, args.point_cloud, args.depth_scale)

        if not face_detected:
            print(f'No face detected in is image')

        if args.face_dense:
            cv2.imwrite(f'{args.folder}/{file}_face_dense.jpg', output_image)

        if args.depth_map:
            depth_map = get_depth_map(input_image, points, triangles.simplices, depth_list)
            cv2.imwrite(f'{args.folder}/{file}_face_depth.jpg', depth_map)

        if args.save_mesh:
            points_3d = get_3D_point_cloud(points, depth_list)
            save_ply_mesh(points_3d,  triangles.simplices, f'{args.folder}/{file}_face_dense.ply')

        if args.render_mesh:
            point_cloud_data = get_3D_point_cloud(points, depth_list, w)
            mesh_image = render_mesh(image, point_cloud_data)
            cv2.imwrite(f'{args.folder}/{file}_face_mesh.jpg', mesh_image)


        if args.mesh_view:
            point_cloud_data = get_3D_point_cloud(points, depth_list, w)
            list_view = render_rotate_mesh(image, point_cloud_data)
            save_images_as_gif(list_view, f'{args.folder}/{file}_face_mesh_view.gif')



if __name__ == '__main__':
    # Example usage:
    main()
