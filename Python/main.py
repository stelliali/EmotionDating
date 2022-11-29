import copy
import datetime
import getopt
import os

import sys
from enum import Enum
from pathlib import Path
import time
from lime import lime_tabular

from sklearn.model_selection import train_test_split
from PIL import Image
import threading
import pandas
import tensorflow as tf
import utils
from playsound import playsound
from AttentionDetector import AttentionDetector
from EmotionTracker import EmotionTracker, Emotion
from FaceDetection import FaceLandMarks
from FeatureSequence import FeatureSequence
from HeadposeDetector import HeadposeDetector
from GazeDetector import GazeDetector
from utils import *
from MainWindow import MainWindow
from matplotlib import pyplot as plt


class RunMode(Enum):
    EXTRACT = 0
    PREDICT = 1
    TRAIN = 3
    EVALUATE = 4
    CREATE_DATASET_ENGAGED = 5
    CREATE_DATASET_NOT_ENGAGED = 6
    GUI = 7


settings = {
    'camera_id': 0,
    'mode': RunMode.GUI,
    'rot_x_weight': 0.0,
    'rot_y_weight': 0.0,
    'rot_z_weight': 0.0,
    'rot_buffer_size': 4,
    'gaze_buffer_size': 4,
    'num_face': 2,
    'max_angle_x': 20,
    'min_angle_x': -20,
    'max_angle_y': 10,
    'min_angle_y': -10,
    'face_draw_offset': np.array([-50, 0, 200]),
    'draw_line_space': 40,
    'gaze_draw_scalar': 100,
    'face_z_axis_multiplier': 500,
    'path_dataset_lmu': Path('assets/lmu-students-engagement-dataset'),
    'path_dataset_students': Path('assets/Student-engagement-dataset-img'),
    'path_load_featureset': Path('assets/featureset.csv'),
    'path_save_featureset': Path('assets/featureset.csv'),
    'path_save_lime_weights': Path('assets/avg_lime_weights.csv'),
    'path_load_model_head_pose': Path('assets/model_head_pose'),
    'path_save_model_head_pose': Path('assets/model_head_pose'),
    'path_load_model_emotions': Path('assets/model_emotion/model_resnet.102-0.66.hdf5'),
    'face_detector_model_path' : Path('assets/model_face/haarcascade_frontalface_default.xml'),
    'evaluation_path_learning' : Path('./images/model_accuracy_and_loss.png'),
    'path_lime_weights_importance':Path('./images'),
    'episodes': 120,
    'batch_size': 5,
    'attention_buffer_size': 10,
    'attention_buffer_marker': 5,
    'dataset_image_cap': 25,
}

def args_to_settings(argv):
    try:
        opts, args = getopt.getopt(argv, 'c:m:h', ['camera=', 'mode=', 'help'])
    except getopt.GetoptError:
        print
        'main.py -c <camera> -h <help>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print
            'main.py -c <camera>'
            sys.exit()
        elif opt in ("-m", "--mode"):
            try:
                settings['mode'] = RunMode[arg]
            except:
                print('{mode} could not be matched'.format(mode = arg))
                sys.exit()

        elif opt in ("-c", "--camera"):
            settings['camera_id'] = int(arg)

    return settings
def draw_face_from_top(img, faces, rotation_vectors):
    img_size = img.shape
    draw_offset = settings['face_draw_offset'] + np.array([img_size[0], 0, 0])

    for p in faces:
        point_offset = np.add(np.subtract(p, faces[234]), draw_offset)
        cv2.circle(img, (int(point_offset[0]), int(point_offset[2])), 3, (255, 255, 255), -1)

    source_point = np.subtract(utils.middle_between(faces[454], faces[234]), faces[234])
    source_point_offset = np.add(source_point, draw_offset)
    source_point_offset_top = (int(source_point_offset[0]), int(source_point_offset[2]))

    rotation_vectors[2][2] = -rotation_vectors[2][2]
    rotation_vectors = [np.add(np.add(v, draw_offset), source_point) for v in rotation_vectors]
    (face_x_axis, face_y_axis, face_z_axis) = rotation_vectors

    cv2.line(img, source_point_offset_top, (int(face_x_axis[0]), int(face_x_axis[2])), (0, 0, 255), 3, -1)
    cv2.line(img, source_point_offset_top, (int(face_y_axis[0]), int(face_y_axis[2])), (0, 255, 0), 3, -1)
    cv2.line(img, source_point_offset_top, (int(face_z_axis[0]), int(face_z_axis[2])), (255, 0, 0), 3, -1)

    cv2.circle(img, source_point_offset_top, 3, (0, 0, 255), -1)

def draw(img, is_distracted, head_rotations, emotion_prob, gaze_directions):
    text_lines = []

    if draw_hud[0]:
        text_lines.append(("X: {0:.2f}".format(head_rotations[0] + settings['rot_x_weight']), (0, 255, 0)))
        text_lines.append(("Y: {0:.2f}".format(head_rotations[1] + settings['rot_y_weight']), (0, 255, 0)))
        text_lines.append(("Z: {0:.2f}".format(head_rotations[2] + settings['rot_z_weight']), (0, 255, 0)))

        try:
            emotion = EmotionTracker.probability_to_emotion(emotion_prob)
            emotion_txt = "Emotion: {}".format(emotion.name)

        except ValueError:
            emotion_txt = "Emotion: Unknown"

        text_lines.append((emotion_txt, (0, 255, 0)))

    # Draws a line for each eyes gaze
    if gaze_directions is not None and draw_feed[0]:
        gaze_scalar = settings['gaze_draw_scalar']

        gaze_left = int(gaze_directions[2][0] * gaze_scalar), int(gaze_directions[2][1] * gaze_scalar)
        gaze_right = int(gaze_directions[2][0] * gaze_scalar), int(gaze_directions[2][1] * gaze_scalar)

        pupil_left = int(gaze_directions[0][0]), int(gaze_directions[0][1])
        pupil_right = int(gaze_directions[1][0]), int(gaze_directions[1][1])

        cv2.line(img, pupil_left, np.add(pupil_left, gaze_left), (252, 0, 255), 2)
        cv2.line(img, pupil_right, np.add(pupil_right, gaze_right), (252, 0, 255), 2)

        if draw_hud[0]:
            text_lines.append(("l_gaze: {0:.2f}, {1:.2f}".format(gaze_directions[2][0], gaze_directions[2][1]), (0, 255, 0)))
            text_lines.append(("r_gaze: {0:.2f}, {1:.2f}".format(gaze_directions[3][0], gaze_directions[3][1]), (0, 255, 0)))

    if is_distracted is not None and is_distracted:
        text_lines.append(('Distracted', (0, 0, 255)))

    # Set bool for gamification part of the UI
    if (is_distracted != True): distracted[0] = False
    else: distracted[0] = True

    for i, line_tuple in enumerate(text_lines):
        cv2.putText(img, line_tuple[0], (20, i * settings['draw_line_space'] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, line_tuple[1], 3)

    cv2.imshow('frame', img)

def sound(attention_buffer,old_attention_buffer):

    if attention_buffer.count(True) > old_attention_buffer.count(True) and attention_buffer.count(True) == settings['attention_buffer_marker']:
        threading.Thread(target=(lambda: playsound('finger_snap.mp3'))).start()

def plot_validation(history):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,12))
    fig.suptitle('Training accuracy and loss')
    ax1.plot(history.history['binary_accuracy'])
    ax1.plot(history.history['val_binary_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'val'], loc='upper left')
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'val'], loc='upper left')

    if not os.path.exists(str(settings['evaluation_path_learning'])):
        Path("images").mkdir(parents=True, exist_ok=True)
        with open(str(settings['evaluation_path_learning']), 'w+'):
            pass
    fig.savefig(str(settings['evaluation_path_learning']))

def create_explainer(df,attention_detector):
    attention_detector.load()
    train_features = np.array(df[['Rot_Head_x', 'Rot_Head_y', 'Rot_Head_z'] + [e.name for e in Emotion] + ['eye_gaze_l_x', 'eye_gaze_l_y', 'eye_gaze_r_x','eye_gaze_r_y']])
    explainer = lime_tabular.LimeTabularExplainer(
    training_data=train_features,
    feature_names=df.columns[:-1],
    class_names=['Not engaged', 'Engaged'],
    mode='classification'
    )
    return explainer

def return_weights(exp):

    """Get weights from LIME explanation object"""

    exp_list = exp.as_map()[1]
    exp_list = sorted(exp_list, key=lambda x: x[0])
    exp_weight = [x[1] for x in exp_list]

    return exp_weight

def get_lime_weights_for_dataset(df, explainer,attention_detector):
    weights = []
    for i in range(df.shape[0]):
        test_vector=df.iloc[i,:-1].values
        test_vector=test_vector.reshape((1,14))
        test_vector=np.reshape(test_vector, 14)
        exp = explainer.explain_instance(test_vector, attention_detector.predict_proba,  num_features=14)
        exp_weight = return_weights(exp)
        weights.append(exp_weight)

    lime_weights = pandas.DataFrame(data=weights,columns=df.columns[:-1])
    return lime_weights

def get_average_lime_weights(lime_weights):
    abs_mean = lime_weights.abs().mean(axis=0)
    abs_mean = pandas.DataFrame(data={'feature':abs_mean.index, 'abs_mean':abs_mean})
    abs_mean = abs_mean.sort_values('abs_mean')
    return abs_mean

def plot_average_weights(abs_mean):
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8,8))

    y_ticks = range(len(abs_mean))
    y_labels = abs_mean.feature
    plt.barh(y=y_ticks,width=abs_mean.abs_mean)

    plt.yticks(ticks=y_ticks,labels=y_labels,size= 15)
    plt.title('LIME Weights for the dataset as feature importance')
    plt.ylabel('')
    plt.xlabel('Mean |Weight|',size=20)
    plt.tight_layout()

    img_path = str(os.path.join(settings['path_lime_weights_importance'], "avg_weights.png"))
    if not os.path.exists(settings['path_lime_weights_importance']):
        Path(settings['path_lime_weights_importance']).mkdir(parents=True, exist_ok=True)
        with open(str(img_path), 'w+'):
            pass

    plt.savefig(img_path)
def extract_features(img, smoothing=False):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_faces, faces = detector.find_face_landmark(copy.copy(img))
    faces_position = face_detector.detectMultiScale(gray_image, 1.3, 5)

    if not faces:
        return None

    faces = np.array(faces[0])
    head_rotations_grad, rotation_vectors = headpose_detector.get_rotation_from_faces(faces, smoothing=smoothing)
    gaze_directions = gaze_detector.get_rotation_from_eyes(copy.copy(img), faces, smoothing=smoothing)
    emotion_prob = emotion_tracker.predict_emotion(copy.copy(img), faces_position)

    head_rotations_deg = [radian_to_degrees(axis) for axis in head_rotations_grad]
    if draw_feed[0]:
        draw_face_from_top(img, faces, rotation_vectors)
    return head_rotations_deg, emotion_prob, gaze_directions

def get_img_from_camera(cap):
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    success, img = cap.read()

    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        return None
    else:
        return img

def create_featureset_from_dataset():
    path_dataset = settings['path_dataset_lmu']
    if(str(path_dataset).__contains__('lmu')):
        images = path_dataset.glob('*/*.jpg')
    else:
        images = path_dataset.glob('*/*/*.jpg')
    feature_set = []

    while cv2.waitKey(1) != ord('q'):
        next_image = next(images, None)
        if next_image is None:
            img = None
        else:
            img = cv2.imread(str(next_image))

        if img is None:
            break

        features = extract_features(img, smoothing=False)
        if features is None:
            continue
        else:
            head_rotations, emotion_prob, gaze_directions = features
        if gaze_directions is None:
            continue

        is_distracted = not str(next_image).__contains__('Engaged')
        df_column = list(head_rotations)
        df_column.extend(emotion_prob)
        df_column.extend(gaze_directions[2:].flatten())
        df_column.append(1 if is_distracted else 0)
        feature_set.append(df_column)

        draw(img, is_distracted, head_rotations, emotion_prob, gaze_directions)

    columns = ['Rot_Head_x', 'Rot_Head_y', 'Rot_Head_z', 'Labels']
    columns[3: 3] = [e.name for e in Emotion]
    columns[10: 10] = ['eye_gaze_l_x', 'eye_gaze_l_y', 'eye_gaze_r_x', 'eye_gaze_r_y']
    df = pandas.DataFrame(feature_set, columns=columns)
    df.to_csv(settings['path_save_featureset'], index=False)

def run_live_prediction():
    capture_stop[0] = False
    cap = cv2.VideoCapture(settings['camera_id'])
    attention_detector = AttentionDetector(settings, input_layer_dim=14, output_layer_dim=1)
    attention_detector.load()
    attention_buffer = [False for i in range(settings['attention_buffer_size'])]
    old_attention_buffer = attention_buffer.copy()

    # Checl key press for normal mode and stop flag for GUI mode
    while (cv2.waitKey(1) != ord('q')) and (capture_stop[0] != True):
        img = get_img_from_camera(cap)

        if img is None:
            break

        features = extract_features(img, smoothing=True)
        if features is None:
            continue
        else:
            head_rotations, emotion_probs, gaze_directions = features

        if gaze_directions is not None:
            feature_vector = list(head_rotations)
            feature_vector.extend(emotion_probs)
            feature_vector.extend(gaze_directions[2:].flatten())
            is_distracted = attention_detector.is_distracted_predict(np.array([feature_vector]))
            attention_buffer.append(is_distracted)
            attention_buffer.pop(0)
            print(f'Distracted: {attention_buffer.count(True)}, Engaged: {attention_buffer.count(False)}')

        draw(img, attention_buffer.count(True) > settings['attention_buffer_marker'], head_rotations, emotion_probs, gaze_directions)
        if (sound_play[0]):
            sound(attention_buffer,old_attention_buffer)
        old_attention_buffer=attention_buffer.copy()
    cap.release()

    # Needed for GUI mode, otherwise no new window can be created
    cv2.destroyAllWindows()

def create_dataset(isEngaged):
    cap = cv2.VideoCapture(settings['camera_id'])
    now = time.time()
    img_counter = 0
    img_taken = 0

    path_engaged = Path(os.path.join(settings['path_dataset_lmu'], Path('Engaged')))
    path_disengaged = Path(os.path.join(settings['path_dataset_lmu'], Path('Not engaged')))

    if not os.path.exists(str(settings['path_dataset_lmu'])):
        path_engaged.mkdir(parents=True, exist_ok=True)
        path_disengaged.mkdir(parents=True, exist_ok=True)

    while cap.isOpened() and cv2.waitKey(1) != ord('q'):
        img_counter += 1
        later = time.time()
        difference = int(later - now)
        img = get_img_from_camera(cap)

        drawn_image = copy.copy(img)
        cv2.putText(drawn_image, f'seconds: {difference}  image nr: {img_taken}', (20, img.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow('frame', drawn_image)

        if(settings['dataset_image_cap'] != img_counter):
            continue

        if(isEngaged):
            img_path = os.path.join(path_engaged,
                                    f'img{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")}.jpg')
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(img_path)
        else:
            img_path = os.path.join(path_disengaged,
                                    f'img{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")}.jpg')
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(img_path)
        img_counter = 0
        img_taken += 1

def train_attention_from_featureset():
    attention_detector = AttentionDetector(settings, input_layer_dim=14, output_layer_dim=1)
    featureset_path = Path(settings['path_load_featureset'])
    df = pandas.read_csv(featureset_path, index_col=False)
    train, test = train_test_split(df, test_size=0.2,random_state=42)
    train_features = np.array(train[['Rot_Head_x', 'Rot_Head_y', 'Rot_Head_z'] + [e.name for e in Emotion] + ['eye_gaze_l_x', 'eye_gaze_l_y', 'eye_gaze_r_x','eye_gaze_r_y']])
    train_targets = tf.convert_to_tensor(train[['Labels']])
    train_sequence = FeatureSequence(train_features, train_targets, settings['batch_size'])
    validation_data = test[['Rot_Head_x', 'Rot_Head_y', 'Rot_Head_z'] + [e.name for e in Emotion] + ['eye_gaze_l_x', 'eye_gaze_l_y', 'eye_gaze_r_x', 'eye_gaze_r_y']].to_numpy()
    validation_features = tf.convert_to_tensor(test[['Rot_Head_x', 'Rot_Head_y', 'Rot_Head_z'] + [e.name for e in Emotion] + ['eye_gaze_l_x', 'eye_gaze_l_y', 'eye_gaze_r_x', 'eye_gaze_r_y']])
    validation_targets = tf.convert_to_tensor(test[['Labels']])
    validation_sequence = FeatureSequence(validation_features, validation_targets, settings['batch_size'])
    history = attention_detector.fit(train_sequence, validation_sequence, settings['batch_size'], settings['episodes'],shuffle=True)
    attention_detector.save()
    plot_validation(history)

def create_gui():
    main_window = MainWindow([run_live_prediction], [capture_stop, sound_play, draw_hud, draw_feed, distracted])

def attention_evaluation():
    attention_detector = AttentionDetector(settings, input_layer_dim=14, output_layer_dim=1)
    featureset_path = Path(settings['path_load_featureset'])
    df = pandas.read_csv(featureset_path, index_col=False)
    explainer=create_explainer(df, attention_detector)
    lime_weights=get_lime_weights_for_dataset(df, explainer,attention_detector)


    lime_weights.to_csv(settings["path_save_lime_weights"])
    abs_mean=get_average_lime_weights(lime_weights)
    plot_average_weights(abs_mean)


def main(argv):
    settings = args_to_settings(argv)
    global headpose_detector
    global face_detector
    global detector
    global emotion_tracker
    global gaze_detector
    global main_window
    global capture_stop, sound_play, draw_hud, draw_feed, distracted
    headpose_detector = HeadposeDetector(settings)
    face_detector = cv2.CascadeClassifier(str(settings['face_detector_model_path']))
    detector = FaceLandMarks(settings)
    emotion_tracker = EmotionTracker(settings)
    gaze_detector = GazeDetector(settings)

    sound_play, draw_hud, draw_feed, distracted = enabled(4)
    capture_stop = [False]

    match settings['mode']:
        case RunMode.PREDICT:
            run_live_prediction()
        case RunMode.EXTRACT:
            create_featureset_from_dataset()
        case RunMode.TRAIN:
            train_attention_from_featureset()
        case RunMode.EVALUATE:
            attention_evaluation()
        case RunMode.CREATE_DATASET_ENGAGED:
            create_dataset(True)
        case RunMode.CREATE_DATASET_NOT_ENGAGED:
            create_dataset(False)
        case RunMode.GUI:
            capture_stop, sound_play, draw_hud, draw_feed = disabled(4)
            create_gui()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1:])
