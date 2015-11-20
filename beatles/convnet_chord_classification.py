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

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization

## Data loading

# song = "The_Beatles/01_-_Please_Please_Me/08_-_Love_Me_Do"
song = "The_Beatles/03_-_A_Hard_Day's_Night/05_-_And_I_Love_Her"

labels_file = 'data/beatles/chord-pcs/4096_2048/'+song+'.pcs'
features_file = 'data/beatles/chromagram/block=4096_hop=2048_bins=-48,67_div=1/'+song+'.npz'

### Chromagram features

data = np.load(features_file)

features = data['X']
times = data['times']

### Chord labels

df_labels = pd.read_csv(labels_file, sep='\t')
labels_pcs = df_labels[df_labels.columns[1:]].as_matrix()

## Data preprocessing

### Features

# scaler = MinMaxScaler()
# X = scaler.fit_transform(features).astype('float32')

# let's rescale the features manually so that the're the same in all songs
# the range (in dB) is -120 to X.shape[1] (= 115)
X = (features.astype('float32') - 120) / (features.shape[1] - 120)

### Labels

label_dict = dict((c,i) for (i, c) in enumerate(sorted(df_labels['label'].drop_duplicates())))
label_classes = df_labels['label'].apply(lambda l: label_dict[l]).as_matrix().reshape(-1, 1)
n_classes = len(label_dict)
label_ohe = OneHotEncoder(n_values=n_classes)
labels_ohe = label_ohe.fit_transform(label_classes).toarray().astype(np.int32)
labels_ohe

### Splitting datasets - training, validation, test

ix_train, ix_test = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
ix_train, ix_valid = train_test_split(ix_train, test_size=0.2/(1-0.2), random_state=42)
X_train, X_valid, X_test = X[ix_train], X[ix_valid], X[ix_test]
y_train, y_valid, y_test = labels_pcs[ix_train], labels_pcs[ix_valid], labels_pcs[ix_test]
y_ohe_train, y_ohe_valid, y_ohe_test = labels_ohe[ix_train], labels_ohe[ix_valid], labels_ohe[ix_test]
y_cls_train, y_cls_valid, y_cls_test = label_classes[ix_train], label_classes[ix_valid], label_classes[ix_test]

for d in [X_train, X_valid, X_test, y_train, y_valid, y_test]:
    print(d.shape)

# reshape for 1D convolution
def conv_reshape(X):
    return X.reshape(X.shape[0], X.shape[1], 1)

X_conv_train = conv_reshape(X_train)
X_conv_valid = conv_reshape(X_valid)

## Model training and evaluation

model_dir = 'data/beatles/models'
os.makedirs(model_dir, exist_ok=True)

def new_model_id():
    return 'model_%s' % arrow.get().format('YYYY-MM-DD-HH-mm-ss')

def save_model(model_id, model):
    arch_file = '%s/%s_arch.yaml' % (model_dir, model_id)
    weights_file = '%s/%s_weights.h5' % (model_dir, model_id)
    print('architecture:', arch_file)
    print('weights:', weights_file)
    open(arch_file, 'w').write(model.to_yaml())
    model.save_weights(weights_file)

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

model = Sequential()

model.add(Convolution1D(10, 3, input_shape=(features.shape[1], 1)))
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

print('training the model')
training_hist = model.fit(X_conv_train, y_train, nb_epoch=100)

save_model(model_id, model)

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

def model_report_multilabel(model_predict, X_train, y_train, X_valid, y_valid):
    def report_dataset(X, y_true, title):
        y_pred = model_predict(X)
        print(title + ' accuracy (exatch match):', accuracy_score(y_true, y_pred))
        print(title + ' hamming score (non-exatch match):', 1 - hamming_loss(y_true, y_pred))

    report_dataset(X_train, y_train, 'training')
    report_dataset(X_valid, y_valid, 'validation')

model_report_multilabel(model.predict_classes, X_conv_train, y_train, X_conv_valid, y_valid)

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

# true labels
plot_labels(labels_pcs, 'true')
plot_labels(labels_pcs, 'exact_true', exact=True)

# predicted labels
labels_pred_full = model.predict_classes(conv_reshape(X))
plot_labels(labels_pred_full, 'pred')
plot_labels(labels_pred_full, 'exact_pred', exact=True)

# difference
plot_labels(labels_pcs - labels_pred_full, 'diff')
plot_labels(labels_pcs - labels_pred_full, 'exact_diff', exact=True)

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
