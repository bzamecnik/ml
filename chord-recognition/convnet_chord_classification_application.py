# Chord classification
#
# The task is to classify chords (or more precisely pitch class sets) based on chromagram features.
#
# We use a single Beatles song with just two chord and silence.
#
# The task is in fact multilabel classification, since each pitch class is generally independent.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arrow
import os
import scipy.signal
import scipy.misc

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import hamming_loss, accuracy_score

from keras.models import model_from_yaml

from tfr.spectrogram import create_window
from tfr.files import load_wav
from tfr.analysis import split_to_blocks
from tfr.reassignment import chromagram

## Load model

model_id = 'model_2016-04-16-20-52-03'
model_dir = '../data/beatles/models/' + model_id
model_arch = model_dir + '/' + model_id + '_arch.yaml'
model_weights = model_dir + '/' + model_id + '_weights.h5'

print('loading model:', model_arch)
model = model_from_yaml(open(model_arch).read())
print('loading model wieghts:', model_weights)
model.load_weights(model_weights)

## Load data

song = "The_Beatles/03_-_A_Hard_Day's_Night/05_-_And_I_Love_Her"

audio_file = '../data/beatles/audio-cd/' + song + '.wav'

### Chromagram features

# labels_file = '../data/beatles/chord-pcs/4096_2048/'+song+'.pcs'
# features_file = '../data/beatles/chromagram/block=4096_hop=2048_bins=-48,67_div=1/'+song+'.npz'

# data = np.load(features_file)

# features = data['X']
# times = data['times']

### Chord labels

# df_labels = pd.read_csv(labels_file, sep='\t')
# labels_pcs = df_labels[df_labels.columns[1:]].as_matrix()

block_size = 4096
hop_size = 2048

print('loading audio:', audio_file)
x, fs = load_wav(audio_file)
print('splitting audio to blocks')
x_blocks, times = split_to_blocks(x, block_size, hop_size)
w = create_window(block_size)
print('computing chromagram')
X_chromagram = chromagram(x_blocks, w, fs, to_log=True)
features = X_chromagram

## Data preprocessing

### Features

print('scaling the input features')
# scaler = MinMaxScaler()
# X = scaler.fit_transform(features).astype('float32')
# TODO: there's a bug: should be + 120 on both places!!!
X = (features.astype('float32') - 120) / (features.shape[1] - 120)

# reshape for 1D convolution
def conv_reshape(X):
    return X.reshape(X.shape[0], X.shape[1], 1)

X_conv = conv_reshape(X)

# visualization
#
# def plot_labels(l, title, fifths=False, resample=True, exact=False):
#     if fifths:
#         l = l[:,np.arange(12)*7 % 12]
#     l = l.T
#
#     # file = model_dir+'/'+model_id+'_'+title+'.png'
#
#     if exact:
#         pass
#         # scipy.misc.imsave(file, l)
#     else:
#         if resample:
#             l = scipy.signal.resample(l, 200, axis=1)
#         plt.figure(figsize=(20, 2))
#         plt.imshow(l, cmap='gray', interpolation='none')
#         plt.tight_layout()
#         plt.show()
#         # plt.savefig(file)

# predicted labels
# labels_pred_full = model.predict_classes(X_conv)
# plot_labels(labels_pred_full, 'pred')
# plot_labels(labels_pred_full, 'exact_pred', exact=True)

# in case of input features with original time order we can apply median filter:
# medfilt(labels_pred_full, (15, 1))

model.compile(class_mode='binary', loss='binary_crossentropy', optimizer='adam')

y_pred = (model.predict(X_conv) >= 0.5).astype(np.int32)

pred_file = '../data/beatles/chord-pcs-predicted/%d_%d/%s/%s.tsv' % (block_size, hop_size, model_id, song)
pred_dir = os.path.dirname(pred_file)
os.makedirs(pred_dir, exist_ok=True)
np.savetxt(pred_file, y_pred, delimiter='\t', fmt='%d')

# def plot_labels_true_pred_diff():
#     def plot2d(x):
#         plt.imshow(scipy.signal.resample(x.T, 200, axis=1), cmap='gray', interpolation='none')
#     plt.figure(figsize=(20, 6))
#     ax = plt.subplot(3,1,1)
#     plot2d(labels_pcs)
#     ax.set_title('true')
#     ax = plt.subplot(3,1,2)
#     plot2d(labels_pred_full)
#     ax.set_title('predicted')
#     ax = plt.subplot(3,1,3)
#     plot2d(labels_pred_full - labels_pcs)
#     ax.set_title('difference')
#     plt.tight_layout()
#     plt.show()
#
# plot_labels_true_pred_diff()
