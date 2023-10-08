import cv2
import mediapipe as mp
import numpy as np

from scipy.spatial import Delaunay
from utils.interpolation import bilinear_interpolation_depth, bilinear_interpolation_triangle
from utils.visualize import plot_mesh, plot_point_cloud
from utils.transform import min_max_scale, get_3D_point_cloud
from utils.post_processing import smoothing_point_cloud
from utils.alignment import frontalize


class FaceMeshGenerator:
    def __init__(self, max_loop = 2, 
                detection_conf = 0.2, 
                face_mesh_dect_conf = 0.2,
                min_tracking_conf = 0.2):
    
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence = detection_conf)
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence = face_mesh_dect_conf, 
                                                    min_tracking_confidence = min_tracking_conf)
        
        self.max_loop = max_loop
        self.view_alignment = np.load('data/view/front-view-lmks.npy')
    

    def generate_face_mesh(self, image, point_cloud = False, depth_scale = False, smoothing = False, frontview = False):
        try:
            points = []
            triangles = []
            depth_list = []
            face_detected = []

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results_detection = self.face_detection.process(image_rgb)

            if not results_detection.detections:
                return image, face_detected, points, depth_list, triangles
            
            for detection in results_detection.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face_detected.append(image[y:h, x:w])
    
                results_landmarks = self.face_mesh.process(image_rgb)
                if not results_landmarks.multi_face_landmarks:
                    continue
                
                for landmarks in results_landmarks.multi_face_landmarks:

                    landmarks_list = [(int(landmark.x * iw), int(landmark.y * ih)) for landmark in landmarks.landmark]
                    depth_list = [landmark.z  for landmark in landmarks.landmark]

                    # if get frontal view face:
                    if frontview:
                        current_view = get_3D_point_cloud(landmarks_list, depth_list)
                        front_view = frontalize(current_view, self.view_alignment)
                        landmarks_list = list(front_view[:, :2])

                    # Perform Delaunay triangulation
                    points = np.array(landmarks_list)
                    triangles = Delaunay(points)

                    for _ in range(self.max_loop):
                        for simplex in triangles.simplices:
                            pt1, pt2, pt3 = points[simplex]

                            tri_center = bilinear_interpolation_triangle(A=pt1, B=pt2, C=pt3)
                            landmarks_list.append(tri_center)
                            triangle_depths = [depth_list[i] for i in simplex]
                            depth_list.append(bilinear_interpolation_depth(triangle_depths))

                        # Perform Delaunay triangulation
                        points = np.array(landmarks_list)
                        triangles = Delaunay(points)

                    if smoothing:
                        point_cloud_data = get_3D_point_cloud(landmarks_list, depth_list)
                        smoothed = smoothing_point_cloud(point_cloud_data)

                        landmarks_list = smoothed[:, :2]
                        depth_list = smoothed[:, 2]
                    
                    # Perform Delaunay triangulation
                    points = np.array(landmarks_list)
                    triangles = Delaunay(points)

                    # scaling depth list
                    if depth_scale:
                        depth_list = min_max_scale(depth_list)
                    else: depth_list = np.asarray(depth_list)

                    if point_cloud: 
                        image = plot_point_cloud(image, points.astype(int), depth_list)
                    else: 
                        image = plot_mesh(image, points.astype(int), triangles.simplices, depth_list)        
                return image, face_detected, points, depth_list, triangles         
        except Exception as e:
            raise e