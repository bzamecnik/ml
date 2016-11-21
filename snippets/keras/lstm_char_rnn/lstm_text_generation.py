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
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
import sys

class Dataset:
    def __init__(self, frame_size=40, hop_size=3):
        self.frame_size = frame_size
        self.hop_size = hop_size

        path = get_file('nietzsche.txt',
            origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
        self.text = open(path).read().lower()
        print('corpus length:', len(self.text))

        chars = sorted(list(set(self.text)))
        self.class_count = len(chars)
        print('total chars:', self.class_count)

        self.le = LabelEncoder().fit(chars)

        self.text_ohe = self.text_to_ohe(self.text)

        def split_to_frames(values, frame_size, hop_size):
            """
            Split to overlapping frames.
            """
            return np.stack(values[i:i + frame_size] for i in
                range(0, len(values) - frame_size + 1, hop_size))

        def split_features_targets(frames):
            """
            Split each frame to features (all but last element)
            and targets (last element).
            """
            frame_size = frames.shape[1]
            X = frames[:, :frame_size - 1]
            y = frames[:, -1]
            return X, y

        # cut the text in semi-redundant sequences of frame_size characters
        self.X, self.y = split_features_targets(split_to_frames(
            self.text_ohe, frame_size + 1, hop_size))

        print('X.shape:', self.X.shape, 'y.shape:', self.y.shape)

    def ohe_to_text(self, text_ohe):
        return self.le_to_text(text_ohe.argmax(axis=1))

    def text_to_ohe(self, text):
        return self.le_to_ohe(self.text_to_le(list(text)))

    def le_to_text(self, text_le):
        return ''.join(self.le.inverse_transform(text_le))

    def text_to_le(self, text):
        return self.le.transform(text)

    def le_to_ohe(self, text_le):
        return to_categorical(text_le, nb_classes=self.class_count)

class Model:
    def __init__(self, dataset):
        self.dataset = dataset

        def create_model(dataset):
            # build the model: a single LSTM
            print('Build model...')
            model = Sequential()
            model.add(LSTM(128, input_shape=(dataset.frame_size, dataset.class_count)))
            model.add(Dense(dataset.class_count))
            model.add(Activation('softmax'))

            optimizer = RMSprop(lr=0.01)
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                optimizer=optimizer)
            return model

        self.model = create_model(self.dataset)

    def fit_with_preview(self):
        # output generated text after each iteration
        preview_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: self.preview())
        self.model.fit(
            self.dataset.X, self.dataset.y,
            batch_size=1000, nb_epoch=60,
            callbacks=[preview_callback])

    def generate_chars(self, seed_text, length, temperature=1.0):
        def sample(preds, temperature):
            # helper function to sample an index from a probability array
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)

        window_ohe = self.dataset.text_to_ohe(seed_text)

        for i in range(length):
            # single data point
            x = window_ohe[np.newaxis]

            probs = self.model.predict(x, verbose=0)[0]
            next_index = sample(probs, temperature)

            yield dataset.le_to_text(next_index)

            next_ohe = dataset.le_to_ohe([next_index])
            window_ohe = np.vstack([window_ohe[1:], next_ohe])

    def preview(self, seed_text=None, length=100):
        if seed_text is None:
            start_index = random.randint(0, len(dataset.text) - dataset.frame_size - 1)
            seed_text = self.dataset.text[start_index:start_index + dataset.frame_size]
        print()
        print('----- Generating with seed: "' + seed_text + '"')

        for temperature in [0.2, 0.5, 1.0, 1.2]:
            print('----- temperature:', temperature)
            for char in self.generate_chars(seed_text, length, temperature):
                sys.stdout.write(char)
                sys.stdout.flush()
            print()

if __name__ == '__main__':
    dataset = Dataset()
    model = Model(dataset)
    model.fit_with_preview()
