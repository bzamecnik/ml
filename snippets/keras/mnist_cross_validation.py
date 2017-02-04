import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
import pandas as pd
from sklearn.model_selection import StratifiedKFold


batch_size = 128
nb_classes = 10
nb_epoch = 20
n_splits = 10

# the data, shuffled and split between train and test sets
(X_cv, y_cv), (X_test, y_test) = mnist.load_data()

X_cv = X_cv.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_cv = X_cv.astype('float32')
X_test = X_test.astype('float32')
X_cv /= 255
X_test /= 255
print(X_cv.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_cv = np_utils.to_categorical(y_cv, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

def create_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    return model

skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
results = pd.DataFrame(columns=['loss', 'acc'])
result_stats = None
for i, indices in enumerate(skf.split(X_cv, y_cv)):
    idx_train, idx_val = indices
    print("Running fold", i + 1, "/", n_splits)
    model = create_model()
    if i == 0:
            model.summary()
    X_train, Y_train = X_cv[idx_train], Y_cv[idx_train]
    X_val, Y_val = X_cv[idx_val], Y_cv[idx_val]
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size, nb_epoch=nb_epoch,
        verbose=0)
    results.loc[i] = model.evaluate(X_val, Y_val, verbose=0)
    result_stats = pd.DataFrame({'mean': results.mean(axis=0), 'std': results.std(axis=0)})
    print(result_stats)

print(results)
