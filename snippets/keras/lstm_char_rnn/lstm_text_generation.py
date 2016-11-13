'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.

Source: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
License: MIT (see https://github.com/fchollet/keras/blob/master/LICENSE)
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
import sys

path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
class_count = len(chars)
print('total chars:', class_count)

le = LabelEncoder().fit(chars)
text_le = le.transform(list(text))
text_ohe = to_categorical(text_le)

def ohe_to_text(text_ohe):
    return ''.join(le.inverse_transform(text_ohe.argmax(axis=1)))

def split_to_frames(values, frame_size, hop_size):
    """
    Split to overlapping frames.
    """
    return np.stack(values[i:i + frame_size] for i in range(0, len(values) - frame_size + 1, hop_size))

def split_features_targets(frames):
    """
    Split each frame to features (all but last element)
    and targets (last element).
    """
    frame_size = frames.shape[1]
    X = frames[:, :frame_size-1]
    y = frames[:, -1]
    return X, y

# cut the text in semi-redundant sequences of frame_size characters
frame_size = 40
hop_size = 3

X, y = split_features_targets(split_to_frames(text_ohe, frame_size+1, hop_size))

print('X.shape:', X.shape, 'y.shape:', y.shape)

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(frame_size, class_count)))
model.add(Dense(class_count))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=1000, nb_epoch=1)

    start_index = random.randint(0, len(text) - frame_size - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        current_context = text_ohe[start_index:start_index + frame_size]
        print('----- Generating with seed: "' + ohe_to_text(current_context) + '"')

        for i in range(100):
            x = current_context[np.newaxis]

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_binary = to_categorical([next_index], nb_classes=class_count)

            current_context = np.vstack([current_context[1:], next_binary])

            sys.stdout.write(le.inverse_transform(next_index))
            sys.stdout.flush()
        print()
