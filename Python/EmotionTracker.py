from enum import Enum

import cv2
import keras.models
import numpy as np
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers, models
from keras.regularizers import l2

class Emotion(Enum):
        ANGRY = 0
        DISGUSTED = 1
        FEARFUL = 2
        HAPPY = 3
        SAD = 4
        SURPRISED = 5
        NEUTRAL = 6

class EmotionTracker:
    def __init__(self, settings):
        self.settings = settings
        self.model = None # = self.resnetwork()
        # self.model.compile(loss='categorical_crossentropy',
        #                    optimizer=Adam(lr=0.0001, decay=1e-6),
        #                    metrics=['accuracy'])
        self.load()

    def one_hot_to_emotion(self, emotion_vector):
        for i, flag in enumerate(emotion_vector):
            if flag:
                return self.emotion_mapping[i]
        return None

    def load(self):
        self.model = keras.models.load_model(self.settings['path_load_model_emotions'])

    def predict_emotion(self, img, face_position, tf=None):
        if face_position is None or len(face_position) < 1:
            return np.zeros(7)
        else:
            face_position = face_position[0]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        emotion_offsets = (20, 40)
        x, y, width, height = face_position
        x_off, y_off = emotion_offsets
        x1, x2, y1, y2 = x - x_off, x + width + x_off, y - y_off, y + height + y_off
        gray_face = img[y1:y2, x1:x2]

        try:
            gray_face = cv2.resize(gray_face, (self.model.input_shape[1:3]))
        except:
            return np.zeros(7)

        gray_face = gray_face.astype('float32')
        gray_face = gray_face / 255.0
        gray_face = gray_face - 0.5
        gray_face = gray_face * 2.0
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        prediction = np.array(self.model.predict(gray_face))
        return prediction[0]

    @staticmethod
    def emotion_to_one_hot_encoding(emotion:Emotion):
        emotion_vector = np.zeros(7)
        emotion_vector[emotion.value] = 1
        return emotion_vector

    @staticmethod
    def probability_to_emotion(probability_vector):
        if np.max(probability_vector) == 0:
            raise ValueError('Emotion probabilities are 0. Can\'t be mapped to emotion')
        emotion_index = np.argmax(probability_vector)
        return Emotion(emotion_index)

    def resnetwork(input_shape, num_classes, l2_regularization=0.01):
        regularization = l2(l2_regularization)

        img_input = Input(input_shape)
        x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                   use_bias=False)(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                   use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # net 1
        residual = Conv2D(16, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(16, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(16, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # net 2
        residual = Conv2D(32, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # net 3
        residual = Conv2D(64, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # net 4
        residual = Conv2D(128, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        x = Conv2D(num_classes, (3, 3),
                   padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        output = Activation('softmax', name='predictions')(x)

        model = Model(img_input, output)
        return model