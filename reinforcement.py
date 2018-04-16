import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.models import Sequential
from keras import initializers, regularizers
from keras.initializers import normal, identity
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, SGD, Adam

from features import *
from metrics import *
import cv2

# Different actions that the agent can do
number_of_actions = 9
# Actions captures in the history vector
actions_of_history = 4
# Visual descriptor size
visual_descriptor_size = 25088
# Reward movement action
reward_movement_action = 1
# Reward terminal action
reward_terminal_action = 3
# IoU required to consider a positive detection
iou_threshold = 0.5
# gamma is the amount of future reward considered
gamma = 0.90

path_vgg = "../vgg16_weights_tf_dim_ordering_tf_kernels.h5"
model_vgg = obtain_compiled_vgg_16(path_vgg)
inputs = [K.learning_phase()] + model_vgg.inputs
_convout1_f = K.function(inputs, [model_vgg.layers[31].output])

def get_target_y(action, xmin, xmax, ymin, ymax, bb_shape, gt_mask, mask, img, model_qn, history_vector):
    """
    This method updates the target y for each of the actions.

    Args:
        action (int): action to move the bounding box
        xmin, ymin, xmax, ymax (int): coordinates of bounding box
        bb_shape (tuple):  bounding box height and width
        gt_mask, mask (numpy array): ground truth and bounding box mask
        img (numpy array): image frame
        model_qn (object): Q-network model
        history_vector (list): list of previous actions
    Returns:
        y (list): y target for the Q-network to learn
    """
    original_shape = ( img.shape[0] , img.shape[1] )
    y = np.zeros([1,9])[0]
    for action in range(1,10):
        (region_mask_tmp, xmin_tmp, xmax_tmp, ymin_tmp, ymax_tmp) = shift_mask(action, xmin, xmax, ymin, ymax, original_shape, bb_shape)
        old_iou, new_iou = get_iou(gt_mask, mask, region_mask_tmp)
        reward = new_iou - old_iou
        region_img_tmp = img[ymin_tmp:ymax_tmp,xmin_tmp:xmax_tmp]
        history_vector = update_history_vector(history_vector, action)
        new_state = get_state(region_img_tmp, history_vector)
        maxQ = np.max( model_qn.predict(new_state.T, batch_size=1) )
        update = (reward + (gamma * maxQ))
        y[action-1] = update
    return y

def update_history_vector(history_vector, action):
    """
    This method updates the history vector with the new action.

    Args:
        history_vector (list): list of previous actions
        action (int): action index
    Returns:
        updated_history_vector (list): list of previous actions
    """
    action_vector = np.zeros(number_of_actions)
    action_vector[action-1] = 1
    size_history_vector = np.size(np.nonzero(history_vector))
    updated_history_vector = np.zeros(number_of_actions*actions_of_history)
    if size_history_vector < actions_of_history:
        aux2 = 0
        for l in range(number_of_actions*size_history_vector, number_of_actions*size_history_vector+number_of_actions - 1):
            history_vector[l] = action_vector[aux2]
            aux2 += 1
        return history_vector
    else:
        for j in range(0, number_of_actions*(actions_of_history-1) - 1):
            updated_history_vector[j] = history_vector[j+number_of_actions]
        aux = 0
        for k in range(number_of_actions*(actions_of_history-1), number_of_actions*actions_of_history):
            updated_history_vector[k] = action_vector[aux]
            aux += 1
        return updated_history_vector

def get_state(region_img, history_vector):
    """
    This method gets the state descriptor from the cropped region and history
    vector.

    Args:
        region_img (numpy array): cropped image
        history_vector (list): list of previous actions
    Returns:
        state (numpy array): one hot vector descriptor
    """
    im = cv2.resize(region_img, (224, 224)).astype(np.float32)
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'th':
        im = im[::-1, :, :]
        im = im.transpose((2, 0, 1))
    else:
        im = im[:, :, ::-1]
        im = im.transpose((1, 0, 2))
    im = np.expand_dims(im, axis=0)
    descriptor_image = _convout1_f([0] + [im])
    descriptor_image = np.reshape(descriptor_image, (visual_descriptor_size, 1))
    history_vector = np.reshape(history_vector, (number_of_actions*actions_of_history, 1) )
    state = np.vstack((descriptor_image, history_vector))
    return state

def get_q_network(weights_path):
    """
    This method returns the Q-network.

    Args:
        weights_path (string): the path to the weights of the Q-network
    Returns:
        model: Q-network
    """
    model = Sequential()
    initializers.VarianceScaling(scale=0.01, mode='fan_in', distribution='normal', seed=None)
    model.add(Dense(1024, kernel_initializer='VarianceScaling',kernel_regularizer=regularizers.l2(0.01), input_shape=(25124,) ))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_actions, kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('linear'))
    adam = Adam(lr=1e-6)
    model.compile(loss='mse', optimizer=adam)
    if weights_path != "0":
        model.load_weights(weights_path)
    return model
