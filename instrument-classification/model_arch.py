from keras.layers import Activation, BatchNormalization, Convolution2D, \
    Dense, Dropout, Flatten, Input, MaxPooling2D, Reshape
from keras.models import Model


def create_model(input_shape, class_count):
    inputs = Input(shape=input_shape)

    # add one more dimension for convolution
    x = Reshape(input_shape + (1, ))(inputs)

    def convolution_block(filter_count, dropout):
        def create(x):
            x = Convolution2D(filter_count, 3, 3, activation='relu')(x)
            x = Convolution2D(filter_count, 3, 3, activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(dropout)(x)
            return x
        return create

    x = convolution_block(filter_count=32, dropout=0.1)(x)
    x = convolution_block(filter_count=64, dropout=0.1)(x)
    x = convolution_block(filter_count=64, dropout=0.1)(x)

    x = Flatten()(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(class_count)(x)
    x = BatchNormalization()(x)
    predictions = Activation('softmax')(x)

    model = Model(inputs, predictions)

    model.compile(loss='categorical_crossentropy', optimizer='adam',
        metrics=['accuracy'])

    return model
