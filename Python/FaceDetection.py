import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp

class FaceLandMarks():
    def __init__(self, settings, staticMode=False,maxFace=1, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFace = maxFace
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.settings = settings

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.maxFace)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    # Facelandmark map can be found here: ./assets/canonical_face_model_uv_visualization.png
    @staticmethod
    def extract_imagepoints(faces):
        return np.array([
            faces[94][0:2],  # Nose tip
            faces[152][0:2],  # Chin
            faces[33][0:2],  # Left eye left corner
            faces[263][0:2],  # Right eye right corne
            faces[78][0:2],  # Left Mouth corner
            faces[308][0:2]  # Right mouth corner
        ], dtype="double")

    def find_face_landmark(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y, z = int(lm.x * iw), int(lm.y * ih), int(lm.z * self.settings['face_z_axis_multiplier'])
                    # cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
                    # print(id, x, y)
                    face.append([x, y, z])
                faces.append(face)
        return img, faces

    @staticmethod
    def get_3d_model_points():
        return np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])