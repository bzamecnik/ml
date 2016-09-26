from keras.models import model_from_json
import math
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import hamming_loss, accuracy_score, roc_auc_score
import sys

sys.path.append('../tools/music-processing-experiments')

from spectrogram import create_window
from files import load_wav
from analysis import split_to_blocks
from reassignment import chromagram

data_dir = '../data'

# load model

model_id = 'model_2016-05-15-19-03-49'
model_dir = data_dir + '/beatles/models/' + model_id
model_arch = '%s/%s_arch.json' % (model_dir, model_id)
model_weights = '%s/%s_weights.h5' % (model_dir, model_id)
print('loading model:', model_arch)
model = model_from_json(open(model_arch).read())
print('loading model weights:', model_weights)
model.load_weights(model_weights)

model.compile(loss='binary_crossentropy', optimizer='adam')

# load data

song = "The_Beatles/03_-_A_Hard_Day's_Night/05_-_And_I_Love_Her"
audio_file = data_dir + '/beatles/audio-cd/' + song + '.wav'

block_size = 4096
hop_size = 2048

print('loading audio:', audio_file)
x, fs = load_wav(audio_file)
print('splitting audio to blocks')
x_blocks, times = split_to_blocks(x, block_size, hop_size)
w = create_window(block_size)
print('computing chromagram')
X_chromagram = chromagram(x_blocks, w, fs, to_log=True)

### Features

# let's rescale the features manually so that the're the same in all songs
# the range (in dB) is -120 to X.shape[1] (= 115)
# TODO: there's a bug: should be + 120 on both places!!!
def normalize(X):
    return (X.astype('float32') - 120) / (X.shape[1] - 120)

X_orig = normalize(X_chromagram)

print(X_orig.shape)

frame_count, feature_count = X_orig.shape
target_count = 12

# we'll cut the datasets into small sequences of frames
max_seq_size = 100

def pad_sequences(a, max_seq_size):
    """
    Cuts the list of frames into fixed-length sequences.
    The end is padded with zeros if needed.
    (frame_count, feature_count) -> (seq_count, max_seq_size, feature_count)
    """
    n = len(a)
    n_padded = max_seq_size * (math.ceil(n / max_seq_size))
    a_padded = np.zeros((n_padded, a.shape[1]), a.dtype)
    a_padded[:n, :] = a
    return a_padded.reshape(-1, max_seq_size, a.shape[1])

X_seq = pad_sequences(X_orig, max_seq_size)

print(X_seq.shape)

# add one more dimension (1 input channel for convolution)
X_seq_conv = X_seq.reshape(X_seq.shape[0], max_seq_size, feature_count, 1)

print(X_seq_conv.shape)

X = X_seq_conv

# prediction

batch_size = 32

# predict probabilities
# input shape: (sequence_count, sequence_size, feature_count, 1)
# output shape: (sequence_count, sequence_size, target_count)
# probabilities to binary classes
Y_pred_proba = model.predict(X, verbose=1, batch_size=batch_size)
Y_pred = (Y_pred_proba >= 0.5).astype(np.int32)
# reshape from sequences back and trim to the original frame count
Y_pred = Y_pred.reshape(-1, target_count)[:frame_count]

pred_file = data_dir + '/beatles/chord-pcs-predicted/%d_%d/%s/%s.tsv' % (block_size, hop_size, model_id, song)
pred_dir = os.path.dirname(pred_file)
os.makedirs(pred_dir, exist_ok=True)
np.savetxt(pred_file, Y_pred, delimiter='\t', fmt='%d')
print('saved predicted labels to: ', pred_file)
