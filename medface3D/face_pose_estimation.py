import numpy as np
import math
import cv2

class FacePoseEstimator:
    def __init__(self) -> None:
        pass

    def get_pose(self, point_cloud):
        centroid = np.mean(point_cloud, axis=0)
        covariance_matrix = np.cov(point_cloud - centroid, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        roll = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        pitch = np.degrees(np.arctan2(-eigenvectors[2, 0], np.sqrt(eigenvectors[2, 1] ** 2 + eigenvectors[2, 2] ** 2)))
        yaw = np.degrees(np.arctan2(eigenvectors[2, 1], eigenvectors[2, 2]))

        return roll, pitch, yaw


    def get_face_pose(self, face_landmarks):
        # Extract relevant facial landmarks
        landmarks = face_landmarks.landmark
        left_eye = landmarks[33]  # Left eye landmark
        right_eye = landmarks[263]  # Right eye landmark
        nose_tip = landmarks[2]  # Nose tip landmark

        # Calculate yaw angle (left-right head rotation)
        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        yaw_angle = math.degrees(math.atan2(dy, dx))

        # Calculate pitch angle (up-down head rotation)
        dz = right_eye.z - left_eye.z
        dy = right_eye.y - left_eye.y
        pitch_angle = math.degrees(math.atan2(dz, dy))

        # Calculate roll angle (tilt of the head)
        dx = right_eye.x - left_eye.x
        dz = right_eye.z - left_eye.z
        roll_angle = math.degrees(math.atan2(dz, dx))
        return yaw_angle, pitch_angle, roll_angle
    

    def get_face_lmks(self, face_landmarks, im_w, im_h):
        face_2d = []
        face_3d = []

        for idx, lm in enumerate(face_landmarks.landmark):
            if idx not in [33, 263, 1, 61, 291, 199]:
                continue
            
            x, y = int(lm.x * im_w), int(lm.y * im_h)

            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
        
        face_2d = np.asarray(face_2d, dtype=np.float64)
        face_3d = np.asarray(face_3d, dtype=np.float64)
        return face_2d, face_3d
    


    def get_face_pose_v2(self, face_landmarks, im_w, im_h):
        # Refer: https://www.youtube.com/watch?v=-toNMaS4SeQ
        face_2d, face_3d = self.get_face_lmks(face_landmarks, im_w, im_h)

        focal_lenght = 1* im_w
        cam_matrix = np.array([[focal_lenght, 0, im_h / 2],
                               [0, focal_lenght, im_h / 2],
                               [0, 0, 1]])
        
        dist_matrix = np.zeros((4,1), dtype=np.float64)

        sucess, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        rmat, _ = cv2.Rodrigues(rot_vec)

        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        x = angles[0] * 360 
        y = angles[1] * 360
        z = angles[2] * 360

        return x, y, z


