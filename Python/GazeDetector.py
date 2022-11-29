import copy

import cv2
import numpy as np
import mediapipe as mp
import camera
from utils import relative, relative_T

class GazeDetector:

    def __init__(self, settings):
        self.settings = settings
        self.dist_coeffs = np.zeros((4, 1)) # Assumes no lens distortion
        self.mp_face_mesh = mp.solutions.face_mesh

        # The center of the eye ball (3D model)
        self.Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
        self.Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])
        self.gaze_buffer = None

    def get_face_mesh(self):
        face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Include iris landmarks in face mesh model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        return face_mesh

    def get_rotation_from_eyes(self, img, faces, smoothing=False):
        """
        This function calculates the eye rotations, normalizes it and draws the gaze direction into the frame.\n
        Result is returned with format: (l_eye_gaze_x, l_eye_gaze_y, r_eye_gaze_x, y_eye_gaze_y)
        """
        frame = img
        face_mesh = self.get_face_mesh()
        with face_mesh:
            # Pass by reference (can improve performance)
            frame.flags.writeable = False

            # RGB conversion for face-mesh model
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(frame)

            if results.multi_face_landmarks:
                points = results.multi_face_landmarks[0]

                (image_points, image_points_T, image_points_faces, image_points_faces1, model_points_broad, model_points_slim) = self.get_image_points(img, points, faces)

                #camera_matrix = self.get_camera_matrix(img)
                camera_matrix = camera.get_camera_matrix_img(img.shape[1], center = (img.shape[1] / 2, img.shape[0] / 2))

                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points_broad, image_points, camera_matrix,
                                                                          self.dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

                # Location of pupil (2D)
                left_pupil = relative(points.landmark[468], img.shape)
                right_pupil = relative(points.landmark[473], img.shape)

                # Transform image point to world point
                _, transformation, _ = cv2.estimateAffine3D(image_points_T, model_points_slim)
                if(transformation is None):
                    _, transformation, _ = cv2.estimateAffine3D(image_points_T, model_points_broad)

                # Check if estimateAffine3D succeeded
                if transformation is not None:
                    # Project pupil image point into 3d world point
                    pupil_world_cord_left = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T
                    pupil_world_cord_right = transformation @ np.array([[right_pupil[0], right_pupil[1], 0, 1]]).T

                    # 3D gaze point (10 is arbitrary value denoting gaze distance)
                    S1 = self.Eye_ball_center_left + (pupil_world_cord_left - self.Eye_ball_center_left) * 10
                    S2 = self.Eye_ball_center_right + (pupil_world_cord_right - self.Eye_ball_center_right) * 10

                    # Project a 3D gaze direction onto image plane
                    (eye_pupil2D_left, _) = cv2.projectPoints((int(S1[0]), int(S1[1]), int(S1[2])), rotation_vector,
                                                         translation_vector, camera_matrix, self.dist_coeffs)
                    (eye_pupil2D_right, _) = cv2.projectPoints((int(S2[0]), int(S2[1]), int(S2[2])), rotation_vector,
                                                         translation_vector, camera_matrix, self.dist_coeffs)

                    # Project 3D head pose into image plane
                    (head_pose_left, _) = cv2.projectPoints((int(pupil_world_cord_left[0]), int(pupil_world_cord_left[1]), int(40)),
                                                       rotation_vector,
                                                       translation_vector, camera_matrix, self.dist_coeffs)
                    (head_pose_right, _) = cv2.projectPoints((int(pupil_world_cord_right[0]), int(pupil_world_cord_right[1]), int(40)),
                                                       rotation_vector,
                                                       translation_vector, camera_matrix, self.dist_coeffs)

                    # Correct gaze value for head rotation
                    gaze_left = left_pupil + (eye_pupil2D_left[0][0] - left_pupil) - (head_pose_left[0][0] - left_pupil)
                    gaze_right = right_pupil + (eye_pupil2D_right[0][0] - right_pupil) - (head_pose_right[0][0] - right_pupil)

                    if smoothing:
                        gaze_left, gaze_right = self.get_avg_gaze_points(np.array([gaze_left, gaze_right]))

                    l_gaze_vctr = gaze_left[0] - left_pupil[0], gaze_left[1] - left_pupil[1]
                    r_gaze_vctr = gaze_right[0] - right_pupil[0], gaze_right[1] - right_pupil[1]

                    # Normalize coordinates with max length value of each vector
                    l_gaze_vctr = l_gaze_vctr[0] / 94, l_gaze_vctr[1] / 94
                    r_gaze_vctr = r_gaze_vctr[0] / 137, r_gaze_vctr[1] / 137

                    # Returns gaze values of left and right eye
                    return np.array([left_pupil, right_pupil, l_gaze_vctr, r_gaze_vctr])

    def get_image_points(self, img, points, faces):
        # 2D image points with (x,y)
        image_points = np.array([
            relative(points.landmark[4], img.shape),  # Nose tip
            relative(points.landmark[152], img.shape),  # Chin
            relative(points.landmark[263], img.shape),  # Left eye left corner
            relative(points.landmark[33], img.shape),  # Right eye right corner
            relative(points.landmark[287], img.shape),  # Left Mouth corner
            relative(points.landmark[57], img.shape)  # Right mouth corner
        ], dtype="double")

        # 2D image points with (x,y,0)
        image_points_T = np.array([
            relative_T(points.landmark[4], img.shape),  # Nose tip
            relative_T(points.landmark[152], img.shape),  # Chin
            relative_T(points.landmark[263], img.shape),  # Left eye, left corner
            relative_T(points.landmark[33], img.shape),  # Right eye, right corner
            relative_T(points.landmark[287], img.shape),  # Left Mouth corner
            relative_T(points.landmark[57], img.shape)  # Right mouth corner
        ], dtype="double")

        image_points_faces = np.array([
            faces[94][0:2],  # Nose tip
            faces[199][0:2],  # Chin
            faces[33][0:2],  # Left eye left corner
            faces[263][0:2],  # Right eye right corner
            faces[62][0:2],  # Left Mouth corner
            faces[308][0:2]  # Right mouth corner
        ], dtype="double")

        image_points_faces1 = np.array([
            np.append(faces[94][0:2], [0]),  # Nose tip
            np.append(faces[199][0:2], [0]), # Chin
            np.append(faces[33][0:2], [0]),  # Left eye left corner
            np.append(faces[263][0:2], [0]),  # Right eye right corner
            np.append(faces[62][0:2], [0]),  # Left Mouth corner
            np.append(faces[308][0:2], [0])   # Right mouth corner
        ], dtype="double")

        # 3D model points.
        model_points_broad = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0, -63.6, -12.5),  # Chin
            (-43.3, 32.7, -26),  # Left eye, left corner
            (43.3, 32.7, -26),  # Right eye, right corner
            (-35.9, -28.9, -24.1),  # Left Mouth corner
            (35.9, -28.9, -24.1)  # Right mouth corner
        ])

        model_points_slim = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0, -63.6, -12.5),  # Chin
            (-43.3, 32.7, -26),  # Left eye, left corner
            (43.3, 32.7, -26),  # Right eye, right corner
            (-29.9, -28.9, -24.1),  # Left Mouth corner
            (29.9, -28.9, -24.1)  # Right mouth corner
        ])

        return image_points, image_points_T, image_points_faces, image_points_faces1, model_points_broad, model_points_slim

    def get_avg_gaze_points(self, gaze_cords):
        if self.gaze_buffer is None:
            shape = (self.settings['gaze_buffer_size'],) + gaze_cords.shape
            self.gaze_buffer = np.zeros(shape)
        self.gaze_buffer = np.append(self.gaze_buffer[1:], [gaze_cords], axis=0)
        gaze_cords_t = self.gaze_buffer.transpose(1, 2, 0)
        avg_gaze = np.array([[np.average(dim) for dim in coord] for coord in gaze_cords_t])
        return tuple([int(avg_gaze[0][0]), int(avg_gaze[0][1])]), tuple([int(avg_gaze[1][0]), int(avg_gaze[1][1])])

    def draw_image_points(self, img, image_points, color):
        img = copy.copy(img)
        for p in image_points:
            point_offset = np.add(np.subtract(p, np.array([0, 0])), np.array([0, 0]))
            cv2.circle(img, (int(point_offset[0]), int(point_offset[1])), 3, color, -1)
        cv2.imshow('frame', img)

    def draw_image_points3D(self, img, image_points, image_points2):
        img = copy.copy(img)
        for p in image_points:
            point_offset = np.add(np.subtract(p, np.array([0, 0, 0])), np.array([0, 0, 0]))
            cv2.circle(img, (int(point_offset[0]), int(point_offset[1])), 3, (255, 255, 255), -1)
        for p in image_points2:
            point_offset = np.add(np.subtract(p, np.array([0, 0, 0])), np.array([0, 0, 0]))
            cv2.circle(img, (int(point_offset[0]), int(point_offset[1])), 3, (255, 0, 255), -1)
        cv2.imshow('frame', img)
