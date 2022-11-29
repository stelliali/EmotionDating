import math
import cv2
from numpy import cos, sin

relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
relative_T = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)

disabled = lambda n: [[False] for _ in range(n)]
enabled = lambda n: [[True] for _ in range(n)]

def radian_to_degrees(rad):
    return float(rad * 180 / math.pi)


def calc_angle(p1, p2):
    try:
        m = (p1[1] - p2[1])/(p1[0] - p2[0])
        ang = int(math.degrees(math.atan(-1/m)))
    except:
        ang = 90
    return ang


def optimize_face(facebox, img):
    face_img = img[facebox[1]: facebox[3], facebox[0]: facebox[2]]
    face_img = cv2.resize(face_img, (128, 128))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    return face_img

import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def middle_between(p1, p2):
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2

def rotate(degrees, vector):
    theta = np.deg2rad(degrees)
    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return np.append(np.dot(rot, vector[[0, 2]]), vector[1])

def translation_vector_from_faces(faces):
    return (faces[454] - faces[234]) / 2 + faces[234]