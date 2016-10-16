"""
Allows to classify musical instrument families from audio clips using a trained
model.

Input:
- audio clip (WAV/FLAC), 2sec, 44100 Hz sampling rate, mono
- model files (architecture, weights)
Ouput: instrument family [brass, guitar, organ, piano, pipe, reed, strings]
"""

import argparse
import keras
from keras.utils import np_utils
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import pandas as pd
import soundfile as sf

jsonpickle_numpy.register_handlers()


class InstrumentClassifier():
    def __init__(self, model_dir):
        self.model_dir = model_dir

        def load_model(arch_file, weights_file):
            """
            Load Keras model from files - YAML architecture, HDF5 weights.
            """
            with open(arch_file) as f:
                model = keras.models.model_from_yaml(f.read())
            model.load_weights(weights_file)
            model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])
            return model

        def load_model_from_dir(model_dir):
            """
            Load Keras model stored into a given directory with some file-name
            conventions. YAML architecture, HDF5 weights.
            """
            return load_model(
                model_dir + '/model_arch.yaml',
                model_dir + '/model_weights.h5')

        self.model = load_model_from_dir(model_dir)

        with open(model_dir + '/preproc_transformers.json', 'r') as f:
            self.instr_family_le, self.scaler, self.ch = \
                jsonpickle.decode(f.read())

    def load_features(self, audio_file):
        def stereo_to_mono(x):
            # stereo to mono
            if len(x.shape) > 1 and x.shape[1] > 1:
                print('Converting stereo to mono')
                x = x.mean(axis=1)
            return x

        def cut_or_pad_to_length(x, duration, fs):
            desired_length = int(round(duration * fs))
            length = len(x)
            diff = length - desired_length
            abs_diff = abs(diff)
            if diff < 0:
                print('Padding')
                # put the short signal in the middle
                pad_before = abs_diff // 2
                pad_after = abs_diff - pad_before
                x = np.lib.pad(x, (pad_before, pad_after), 'constant')
            elif diff > 1:
                print('Cutting')
                # cut the beginning
                x = x[0:desired_length]
            return x

        def adjust_input(x, fs):
            x = stereo_to_mono(x)
            x = cut_or_pad_to_length(x, 2.0, fs)
            return x

        x, fs = sf.read(audio_file)
        x = adjust_input(x, fs)

        x_chromagram = self.ch.transform(x)
        x_features = self.scaler.transform(x_chromagram.reshape(1, -1)) \
            .reshape(1, *x_chromagram.shape)
        return x_features

    def predict_class_label(self, audio_file):
        x_features = self.load_features(audio_file)
        instrument_class = np_utils.probas_to_classes(self.model.predict(x_features, verbose=0))[0]
        label = self.instr_family_le.inverse_transform(instrument_class)
        return label

    def predict_probabilities(self, audio_file):
        x_features = self.load_features(audio_file)
        proba = self.model.predict(x_features, verbose=0).flatten()
        df = pd.DataFrame({
            'probability': proba,
            'class': np.arange(len(proba)),
            'label': self.instr_family_le.classes_})
        df.sort_values('probability', ascending=False, inplace=True)
        df.set_index('class', inplace=True)
        return df

    def class_label_from_probabilities(self, probabilities):
        return probabilities.iloc[0]['label']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Classifies music instrument family from an audio clip.')
    parser.add_argument('audio_file', metavar='AUDIO_FILE', type=str,
        help='audio file (WAV, FLAC)')
    parser.add_argument('-m', '--model-dir', type=str,
        help='directory with model architecture, weights and preprocessing transformers')
    parser.add_argument('-p', '--proba', action='store_true', default=False,
        help='print probabilities, not just class')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = InstrumentClassifier(args.model_dir)
    if args.proba:
        print(model.predict_probabilities(args.audio_file))
    else:
        print(model.predict_class_label(args.audio_file))
