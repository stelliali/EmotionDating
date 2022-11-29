import cv2
import numpy as np

import utils
from FaceDetection import FaceLandMarks
from camera import get_camera_matrix_img
from utils import radian_to_degrees


class HeadposeDetector:
    def __init__(self, settings):
        self.settings = settings
        self.faces_buffer = None
        self.dist_coeffs = np.zeros((4, 1))
        self.model_points = FaceLandMarks.get_3d_model_points()

    def get_head_rotation_solvePNP(self, img, image_points, smoothing=False):
        if smoothing: image_points = self.get_avg_img_points(image_points)

        focal_length = img.shape[1]
        center = (img.shape[1] / 2, img.shape[0] / 2)
        camera_matrix = get_camera_matrix_img(focal_length, center)

        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points,
                                                                      image_points, camera_matrix,
                                                                      self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, self.dist_coeffs)
        nose_end_point2D = nose_end_point2D[0, 0]

        # Rotation Vector is in radians.... Convert to degrees
        rot_x = (radian_to_degrees(rotation_vector[0])) % 360
        rot_y = (radian_to_degrees(rotation_vector[1])) % 360
        rot_z = (radian_to_degrees(rotation_vector[2])) % 360

        return (rot_x, rot_y, rot_z), nose_end_point2D

    def get_rotation_from_faces(self, faces, smoothing=False):
        if smoothing: faces = self.get_avg_img_points(faces)

        face_x_axis = np.subtract(faces[454], faces[234])
        face_y_axis = np.subtract(faces[10], faces[152])
        face_z_axis = np.array(face_x_axis)[[2, 1, 0]]

        rotation_vectors = (face_x_axis, face_y_axis, face_z_axis)
        rotation_vectors = [v / np.linalg.norm(v) * 100 for v in rotation_vectors]

        rot_x = utils.angle_between(face_y_axis[[1, 2]], [-1, 0])
        rot_y = utils.angle_between(face_x_axis[[0, 2]], [1, 0])
        rot_z = utils.angle_between(face_x_axis[[0, 1]], [1, 0])

        return (rot_x, rot_y, rot_z), rotation_vectors

    def get_avg_img_points(self, faces):
        if self.faces_buffer is None:
            shape = (self.settings['rot_buffer_size'],) + faces.shape
            self.faces_buffer = np.zeros(shape)
        self.faces_buffer = np.append(self.faces_buffer[1:], [faces], axis=0)
        faces_t = self.faces_buffer.transpose(1, 2, 0)
        return np.array([[np.average(dim) for dim in coord] for coord in faces_t])