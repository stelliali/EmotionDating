import numpy as np

def get_camera_matrix_csv():
    K = np.loadtxt("assets/cameraMatrix.csv", delimiter=",", usecols=range(3))
    return K

def get_camera_matrix_img(focal_length, center):
    return np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype='double'
    )

def getDistortionValues():
    distortion = np.loadtxt("assets/distortion_coef.csv", delimiter=",", usecols=range(5))
    return distortion