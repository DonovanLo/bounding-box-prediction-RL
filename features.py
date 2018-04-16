import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import backend as K
K.set_image_dim_ordering('tf')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

import numpy as np

def shift_mask(action, xmin, xmax, ymin, ymax, original_shape, bb_shape):
    """
    This method shifts the bounding box by the given action and returns
    the new bounding box and its mask

    Args:
        action (int): index for the 9 actions to move the bounding box
        xmin (int): x-axis minimum
        xmax (int): x-axis maximum
        ymin (int): y-axis minimum
        ymax (int): y-axis maximum
        original_shape (tuple): image height and image width
        bb-shape (tupe): bounding box height and width

    Returns:
        mask (numpy array): mask with the new bounding box assigned
        xmin, xmax, ymin, ymax (int)
    """
    offset_size = 50
    img_height, img_width = original_shape
    bb_height, bb_width = bb_shape

    if action == 1:         # UP
        ymin = ymin - offset_size
        ymax = ymax - offset_size
    elif action == 2:       # UP-RIGHT
        ymin = ymin - offset_size
        ymax = ymax - offset_size
        xmin = xmin + offset_size
        xmax = xmax + offset_size
    elif action == 3:       # RIGHT
        xmin = xmin + offset_size
        xmax = xmax + offset_size
    elif action == 4:       # DOWN-RIGHT
        ymin = ymin + offset_size
        ymax = ymax + offset_size
        xmin = xmin + offset_size
        xmax = xmax + offset_size
    elif action == 5:       # DOWN
        ymin = ymin + offset_size
        ymax = ymax + offset_size
    elif action == 6:       # DOWN-LEFT
        ymin = ymin + offset_size
        ymax = ymax + offset_size
        xmin = xmin - offset_size
        xmax = xmax - offset_size
    elif action == 7:       # LEFT
        xmin = xmin - offset_size
        xmax = xmax - offset_size
    elif action == 8:       # UP-LEFT
        ymin = ymin - offset_size
        ymax = ymax - offset_size
        xmin = xmin - offset_size
        xmax = xmax - offset_size

    # Correction for action that forces bboxes out of frame
    if ymax >= img_height:
        ymax = img_height
        ymin = img_height - bb_height
    if ymin <= 0:
        ymin = 0
        ymax = bb_height
    if xmax >= img_width:
        xmax = img_width
        xmin = img_width - bb_width
    if xmin <= 0:
        xmin = 0
        xmax = bb_width

    mask = np.zeros(original_shape)
    mask[ymin:ymax,xmin:xmax] = 1
    return mask, xmin, xmax, ymin, ymax

def get_annotation(info, index):
    """
    This method extracts the bounding box object members.

    Args:
        info (object): bounding box object
        index (int): index of the item extracted
    Returns:
        annotation (list): list of bounding box information
    """
    annotation = np.zeros([5], dtype=np.int)
    annotation[0] = int(info.iloc[ index ].frame)
    annotation[1] = int(info.iloc[ index ].xmin)
    annotation[2] = int(info.iloc[ index ].xmax)
    annotation[3] = int(info.iloc[ index ].ymin)
    annotation[4] = int(info.iloc[ index ].ymax)
    return annotation

def obtain_compiled_vgg_16(vgg_weights_path):
    """
    This method calls a vgg16 model.

    Args:
        vgg_weights_path (string): path of the vgg weights
    Returns:
        model (object): vgg model
    """
    model = vgg_16(vgg_weights_path)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model

def vgg_16(weights_path=None):
    """
    This method creates the vgg16 model.

    Args:
        weights_path (string): path of the vgg weights
    Returns:
        model (object): vgg model
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.summary()

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model
