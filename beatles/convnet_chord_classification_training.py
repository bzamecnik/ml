# Chord classification
#
# The task is to classify chords (or more precisely pitch class sets) based on chromagram features.
#
# We use a the whole Beatles dataset (ie. many songs).
#
# The task is in fact multilabel classification, since each pitch class is generally independent.

import numpy as np
import pandas as pd
import matplotlib as mpl
# do not use Qt/X that require $DISPLAY
mpl.use('Agg')
import matplotlib.pyplot as plt
import arrow
import os
import scipy.signal
import scipy.misc

from sklearn.metrics import hamming_loss, accuracy_score

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

## Data loading

dataset_file = 'data/beatles/ml_dataset/chromagram_block=4096_hop=2048_bins=-48,67_div=1/dataset.npz'
dataset = np.load(dataset_file)

X_train, Y_train, X_valid, Y_valid, X_test, Y_test = \
    dataset['X_train'], dataset['Y_train'], \
    dataset['X_valid'], dataset['Y_valid'], \
    dataset['X_test'], dataset['Y_test']

## Data preprocessing

### Features

# scaler = MinMaxScaler()
# X = scaler.fit_transform(features).astype('float32')

# let's rescale the features manually so that the're the same in all songs
# the range (in dB) is -120 to X.shape[1] (= 115)
def normalize(X):
    return (X.astype('float32') - 120) / (X.shape[1] - 120)

X_train = normalize(X_train)
X_valid = normalize(X_valid)
X_test = normalize(X_test)

for d in [X_train, X_valid, X_test, Y_train, Y_valid, Y_test]:
    print(d.shape)

# reshape for 1D convolution
def conv_reshape(X):
    return X.reshape(X.shape[0], X.shape[1], 1)

X_conv_train = conv_reshape(X_train)
X_conv_valid = conv_reshape(X_valid)
X_conv_test = conv_reshape(X_test)

## Model training and evaluation

def new_model_id():
    return 'model_%s' % arrow.get().format('YYYY-MM-DD-HH-mm-ss')

def save_model_arch(model_id, model):
    arch_file = '%s/%s_arch.yaml' % (model_dir, model_id)
    print('architecture:', arch_file)
    open(arch_file, 'w').write(model.to_yaml())

def weights_file(model_id):
    return '%s/%s_weights.h5' % (model_dir, model_id)

def report_model_parameters(model):
    print('number of parameters:', model.count_params())
    print('weights:', [w.shape for w in model.get_weights()])

# #### Notes
#
# - the last layer has to be sigmoid, not softmax
#   - since each output label should be independent a multiple can be active at the same time
# - very sparse inputs can easily saturate sigmoid activation if it's near the first layer
# - class_mode='binary' for multi-label classification
# - predict_classes() then returns a binary vector
# - loss: MAE or binary_crossentropy?
#   - why binary_crossentropy gives worse accuracy than MAE?
#   - binary_crossentropy works ok
# - problems with loss going to NAN after the first training iteration
#   - optimizer clipnorm doesn't help
#   - BatchNormalization doesn't help
#     - BatchNormalization between convolution and activation works
# - BatchNormalization might be useful
# - be aware to use scaled inputs, not raw ones

model_id = new_model_id()
print('model id:', model_id)

model_dir = 'data/beatles/models/' + model_id
os.makedirs(model_dir, exist_ok=True)

model = Sequential()

model.add(Convolution1D(10, 3, input_shape=(X_train.shape[1], 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution1D(10, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling1D(2, 2))

model.add(Flatten())

model.add(Dense(20))
model.add(Activation('relu'))

model.add(Dense(20))
model.add(Activation('relu'))

model.add(Dense(12))
model.add(Activation('sigmoid'))

report_model_parameters(model)

print('compiling the model')
model.compile(class_mode='binary', loss='binary_crossentropy', optimizer='adam')

save_model_arch(model_id, model)

print('training the model')
checkpointer = ModelCheckpoint(filepath=weights_file(model_id), verbose=1, save_best_only=True)

training_hist = model.fit(X_conv_train, Y_train, nb_epoch=10, batch_size=512, callbacks=[checkpointer],
                         verbose=1, validation_data=(X_conv_valid, Y_valid))

def report_training_curve(training_hist):
    losses = training_hist.history['loss']
    print('last loss:', losses[-1])
    plt.figure()
    plt.plot(losses)
    plt.xlabel('epochs')
    plt.ylabel('training loss')
    plt.title('%s - training curve' % model_id)
    plt.suptitle('last loss: %s' % losses[-1])
    plt.savefig(model_dir+'/'+model_id+'_training_losses.png')
    pd.DataFrame({'training_loss': losses}).to_csv(model_dir+'/'+model_id+'_training_losses.tsv', index=None)

report_training_curve(training_hist)

def model_report_multilabel(model_predict, X_train, Y_train, X_valid, Y_valid):
    def report_dataset(X, y_true, title):
        y_pred = model_predict(X)
        print(title + ' accuracy (exatch match):', accuracy_score(y_true, y_pred))
        print(title + ' hamming score (non-exatch match):', 1 - hamming_loss(y_true, y_pred))

    report_dataset(X_train, Y_train, 'training')
    report_dataset(X_valid, Y_valid, 'validation')

model_report_multilabel(model.predict_classes, X_conv_train, Y_train, X_conv_valid, Y_valid)

# visualization

def plot_labels(l, title, fifths=False, resample=True, exact=False):
    if fifths:
        l = l[:,np.arange(12)*7 % 12]
    l = l.T

    file = model_dir+'/'+model_id+'_'+title+'.png'

    if exact:
        scipy.misc.imsave(file, l)
    else:
        if resample:
            l = scipy.signal.resample(l, 200, axis=1)
        plt.figure(figsize=(20, 2))
        plt.imshow(l, cmap='gray', interpolation='none')
        plt.tight_layout()
        plt.savefig(file)

# # true labels
# plot_labels(labels_pcs, 'true')
# plot_labels(labels_pcs, 'exact_true', exact=True)
#
# # predicted labels
# labels_pred_full = model.predict_classes(conv_reshape(X))
# plot_labels(labels_pred_full, 'pred')
# plot_labels(labels_pred_full, 'exact_pred', exact=True)
#
# # difference
# plot_labels(labels_pcs - labels_pred_full, 'diff')
# plot_labels(labels_pcs - labels_pred_full, 'exact_diff', exact=True)

# plot_labels(labels_pred_full[:100], resample=False)
# plot_labels(labels_pcs[:100] - labels_pred_full[:100], resample=False)

# in case of input features with original time order we can apply median filter:
# medfilt(labels_pred_full, (15, 1))

def plot_labels_true_pred_diff():
    def plot2d(x):
        plt.imshow(scipy.signal.resample(x.T, 200, axis=1), cmap='gray', interpolation='none')
    plt.figure(figsize=(20, 6))
    ax = plt.subplot(3,1,1)
    plot2d(labels_pcs)
    ax.set_title('true')
    ax = plt.subplot(3,1,2)
    plot2d(labels_pred_full)
    ax.set_title('predicted')
    ax = plt.subplot(3,1,3)
    plot2d(labels_pred_full - labels_pcs)
    ax.set_title('difference')
    plt.tight_layout()
