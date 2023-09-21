import os
import cv2
import argparse


from mediapipe_facemesh.face_mesh_restructure import FaceMeshRestructure
from utils.save_mesh import save_ply_mesh
from utils.visualize import get_depth_map


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',  '--folder',      default='data/sample' , help='path to sample folder image')
    parser.add_argument('-l',  '--max_loop',    type=int, default=2 ,   help='max loop to reconstruct 3D face mesh')
    parser.add_argument('-p',  '--point_cloud', action="store_true",    help='get image result as point cloud')
    parser.add_argument('-d',  '--depth_scale', action="store_true",    help='depth scale for depth value')
    parser.add_argument('-sv', '--save_mesh',   action="store_true",    help='save result as .ply file')
    parser.add_argument('-dm', '--depth_map',   action="store_true",    help='save result as depth map')
    parser.add_argument('-sf', '--face_dense',  action="store_true",    help='save result as dense map')
    args = parser.parse_args()

    files = os.listdir(args.folder) 

    for _, file in enumerate(files):
        if file.split('.')[-1] not in ['jpg', 'png']:
            continue

        path_read = f'{args.folder}/{file}'
        image = cv2.imread(path_read)

        detector = FaceMeshRestructure(max_loop = args.max_loop)
        input_image = image.copy()
        output_image, face_detected, points, depth_list, triangles = detector.generate_face_mesh(input_image, args.point_cloud, args.depth_scale)
        
        if args.face_dense:
            cv2.imwrite(f'{args.folder}/{file}_face_dense.jpg', output_image)

        if args.depth_map:
            depth_map = get_depth_map(input_image, points, triangles.simplices, depth_list)
            cv2.imwrite(f'{args.folder}/{file}_face_depth.jpg', depth_map)

        if args.save_mesh:
            points_3d = []
            for i, (x,y) in enumerate(points):
                z = 450 / 2 - depth_list[i] * 450
                points_3d.append([x, y, z])
            save_ply_mesh(points_3d,  triangles.simplices, f'{args.folder}/{file}_face_dense.ply')


if __name__ == '__main__':
    # Example usage:
    main()
