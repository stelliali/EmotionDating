from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import Sequential, activations
from keras.layers import Dense, Dropout


class AttentionDetector:
    def __init__(self,
                 settings,
                 input_layer_dim,
                 output_layer_dim,
                 path_dataset=Path('assets/Student-engagement-dataset-img')):
        self.settings = settings
        self.dataset = path_dataset
        self.model = Sequential([
            Dense(16, input_dim=input_layer_dim, activation=activations.relu),
            Dropout(0.2),
            Dense(32, activation=activations.relu),
            Dropout(0.2),
            Dense(output_layer_dim, activation=activations.sigmoid),
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.FalseNegatives()
            ])

    def fit(self,
            training_sequence,
            validation_sequence,
            batch_size,
            epoches,
            shuffle=False):
        print("Fit model on training data")
        return self.model.fit(training_sequence,
                              batch_size=batch_size,
                              epochs=epoches,
                              validation_data=validation_sequence,
                              shuffle=False)

    def save(self):
        self.model.save(self.settings['path_save_model_head_pose'])

    def load(self):
        self.model = tf.keras.models.load_model(
            self.settings['path_load_model_head_pose'])

    def is_distracted_predict(self, features):
        prediction = np.array(self.model.predict(features)).flatten()[0]
        return prediction > 0.5

    def is_distracted_rule_based(self, head_rotations: tuple):
        return not (self.settings['min_angle_x'] < head_rotations[0] < self.settings['max_angle_x']) or \
               not (self.settings['min_angle_y'] < head_rotations[1] < self.settings['max_angle_y'])

    def predict_proba(self, features):
        prediction = self.model.predict(features)
        one = np.ones(prediction.shape)
        prob_other_class = one - prediction
        prob_two_classes = np.concatenate((prediction, prob_other_class),
                                          axis=1)

        return prob_two_classes