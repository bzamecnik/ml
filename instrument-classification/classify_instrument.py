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
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()
import numpy as np
import pandas as pd
import soundfile as sf

from preprocessing import ChromagramTransformer


class InstrumentClassifier():
    def __init__(self, model_dir, preproc_transformers, chromagram_transformer):
        self.model_dir = model_dir
        self.preproc_transformers = preproc_transformers
        self.chromagram_transformer = chromagram_transformer

        def load_model(arch_file, weights_file):
            """
            Load Keras model from files - YAML architecture, HDF5 weights.
            """
            with open(arch_file) as f:
                model = keras.models.model_from_yaml(f.read())
            model.load_weights(weights_file)
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

        with open(preproc_transformers, 'r') as f:
            self.instr_family_le, self.instr_family_oh, self.scaler = \
                jsonpickle.decode(f.read())

        with open(chromagram_transformer, 'r') as f:
            self.ch = ChromagramTransformer(**jsonpickle.decode(f.read()))

    def load_features(self, audio_file):
        x, fs = sf.read(audio_file)
        x_chromagram = self.ch.transform(x)
        x_features = self.scaler.transform(x_chromagram.reshape(1, -1)) \
            .reshape(1, x_chromagram.shape[0], x_chromagram.shape[1], 1)
        return x_features

    def predict_class_label(self, audio_file):
        x_features = self.load_features(audio_file, self.ch, self.scaler)
        instrument_class = self.model.predict_classes(x_features, verbose=0)[0]
        label = self.instr_family_le.inverse_transform(instrument_class)
        return label

    def predict_probabilities(self, audio_file):
        x_features = self.load_features(audio_file)
        proba = self.model.predict_proba(x_features, verbose=0).flatten()
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
        help='directory with model architecture and weights')
    parser.add_argument('-p', '--preproc-transformers', type=str,
        help='file with preprocessing transformer parameters')
    parser.add_argument('-c', '--chromagram-transformer', type=str,
        help='file with chromagram transformer parameters')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = InstrumentClassifier(args.model_dir, args.preproc_transformers,
        args.chromagram_transformer)
    print(model.predict_class_label(args.audio_file))
