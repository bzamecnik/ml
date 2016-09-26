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
import numpy as np
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()
import soundfile as sf

import sys

sys.path.append('../tools/music-processing-experiments/')

from analysis import split_to_blocks
from spectrogram import create_window
from reassignment import chromagram


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
    return load_model(model_dir + '/model_arch.yaml', model_dir + '/model_weights.h5')

def load_features(audio_file, scaler):
    def compute_chromagram(x, sample_rate,
        block_size=4096, hop_size=2048, bin_range=[-48, 67], bin_division=1):
        window = create_window(block_size)
        x_blocks, x_times = split_to_blocks(x, block_size, hop_size, sample_rate)
        return chromagram(x_blocks, window, sample_rate, to_log=True,
            bin_range=bin_range, bin_division=bin_division)

    x_chromagram = compute_chromagram(*sf.read(audio_file))
    x_features = scaler.transform(x_chromagram.reshape(1, -1)).reshape(1, x_chromagram.shape[0], x_chromagram.shape[1], 1)
    return x_features

def predict(model, x_features):
    return model.predict_classes(x_features, verbose=0)[0]

def classify_instrument(audio_file, model_dir, preproc_file):
    model = load_model_from_dir(model_dir)

    with open(preproc_file, 'r') as f:
        instr_family_le, instr_family_oh, scaler = jsonpickle.decode(f.read())

    x_features = load_features(audio_file, scaler)
    instrument_class = predict(model, x_features)
    instrument_label = instr_family_le.inverse_transform(instrument_class)
    return instrument_label

def parse_args():
    parser = argparse.ArgumentParser(description='Classifies music instrument family from an audio clip.')
    parser.add_argument('audio_file', metavar='AUDIO_FILE', type=str,
        help='audio file (WAV, FLAC)')
    parser.add_argument('-m', '--model-dir', type=str,
        help='directory with model architecture and weights')
    parser.add_argument('-p', '--preproc_transformers', type=str,
        help='file with preprocessing transformer parameters')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(classify_instrument(args.audio_file, args.model_dir, args.preproc_transformers))
