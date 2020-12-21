from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Lambda
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
import keras

# The trained classifier is loaded.
# Rewrite this function if you want to use another model architecture than our modified AlexNET.
# A model, which provides a 'predict' function, has to be returned.
def load_classifier(path, img_shape):
    original = keras.models.load(path)
    classifier = build_classifier(img_shape)

    counter = 0
    for layer in original.layers:
        assert (counter < len(classifier.layers))
        classifier.layers[counter].set_weights(layer.get_weights())
        counter += 1

    classifier.summary()

    return classifier


def build_classifier(img_shape):
    input = Input(shape=img_shape)

    # 1st Convolutional Layer
    x = Conv2D(filters=96,
               kernel_size=(11, 11),
               strides=(4, 4),
               padding='valid')(input)
    x = Activation('relu')(x)
    # Pooling
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    # Batch Normalisation before passing it to the next layer
    x = BatchNormalization()(x, training=False)

    # 2nd Convolutional Layer
    x = Conv2D(filters=256,
               kernel_size=(11, 11),
               strides=(1, 1),
               padding='valid')(x)
    x = Activation('relu')(x)
    # Pooling
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    # Batch Normalisation
    x = BatchNormalization()(x, training=False)

    # 3rd Convolutional Layer
    x = Conv2D(filters=384,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='valid')(x)
    x = Activation('relu')(x)
    # Batch Normalisation
    x = BatchNormalization()(x, training=False)

    # 4th Convolutional Layer
    x = Conv2D(filters=384,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='valid')(x)
    x = Activation('relu')(x)
    # Batch Normalisation
    x = BatchNormalization()(x, training=False)

    # 5th Convolutional Layer
    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='valid')(x)
    x = Activation('relu')(x)
    # Pooling
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    # Batch Normalisation
    x = BatchNormalization()(x, training=False)

    # Passing it to a dense layer
    x = Flatten()(x)
    # 1st Dense Layer
    x = Dense(4096, input_shape=img_shape)(x)
    x = Activation('relu')(x)
    # Add Dropout to prevent overfitting
    x = Dropout(0.4)(x, training=False)
    # Batch Normalisation
    x = BatchNormalization()(x, training=False)

    # 2nd Dense Layer
    x = Dense(4096)(x)
    x = Activation('relu')(x)
    # Add Dropout
    x = Dropout(0.4)(x, training=False)
    # Batch Normalisation
    x = BatchNormalization()(x, training=False)

    # 3rd Dense Layer
    x = Dense(1000)(x)
    x = Activation('relu')(x)
    # Add Dropout
    x = Dropout(0.4)(x, training=False)
    # Batch Normalisation
    x = BatchNormalization()(x, training=False)
    x = Dense(2)(x)
    x = Activation('softmax')(x)

    return Model(input, x)