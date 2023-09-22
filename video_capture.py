import os
import cv2
import argparse
import numpy as np

from mediapipe_facemesh.face_mesh_restructure import FaceMeshRestructure
from utils.save_mesh import save_ply_mesh
from utils.visualize import get_depth_map
from utils.render_mesh import render_mesh


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',  '--video_path',  default=0 , help='path to sample folder image')
    parser.add_argument('-l',  '--max_loop',    type=int, default=0 ,   help='max loop to reconstruct 3D face mesh')

    args = parser.parse_args()

    detector = FaceMeshRestructure(max_loop = args.max_loop)
    cap = cv2.VideoCapture(args.video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 3, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if args.video_path == 0:
        name_save = 'local.mp4'
    else: name_save = os.path.basename(args.video_path)

    out = cv2.VideoWriter(f'data/demo/mesh_{name_save}', fourcc, 10, size)


    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if not ret:
            print("Error: Could not read frame.")
            break
        
        input_image = frame
        output_image, face_detected, points, depth_list, triangles = detector.generate_face_mesh(input_image, True, False)
        
        points_3d = []
        for i, (x,y) in enumerate(points):
            z = 450 / 2 - depth_list[i] * 450
            points_3d.append([x, y, z])
        
        point_cloud_data = np.asarray(points_3d)
        mesh_blank = render_mesh(input_image, point_cloud_data)
        mesh_map = render_mesh(input_image, point_cloud_data, False)
        output_image = cv2.hconcat([frame, mesh_blank])
        output_image = cv2.hconcat([output_image, mesh_map])

        out.write(output_image)
        # Display the frame in a window
        cv2.imshow("Video Capture", output_image)

        # Exit the loop when the 'q' key is pressed (or any other key)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close the OpenCV window
    cap.release()
    out.release()
    cv2.destroyAllWindows()



    


if __name__ == '__main__':
    # Example usage:
    main()
