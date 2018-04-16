# Usage:
#       python training.py -t=../PS12_1_7_frames/ -gt=../PS12_1_7_gt.txt
#
import numpy as np
import os, sys
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.preprocessing import image
import random
import argparse

from reinforcement import *
from features import *

import datetime as dt
import pandas as pd

if __name__ == "__main__":
    #### PATHS ####
    path_model = "./model_qn_weights"
    path_vgg = "./vgg16_weights_tf_dim_ordering_tf_kernels.h5"

    #### PARAMETERS ####
    # How many steps to find the target
    epoch_size = 20
    number_of_steps = 20
    epsilon = 1
    batch_size = 200
    frames_step = 30
    buffer_experience_replay = 600
    replay = []
    p = 0               # pointer to store last experience

    # Parse all script arguments
    parser = argparse.ArgumentParser(description='This python script trains on the image frames in the training folder and annotations in the ground-truth folder')
    parser.add_argument("-n", help='the epoch number, if you choose to load an existing model.', metavar='N', type=int, default=0)
    parser.add_argument('-t','--train', required=True, help='Path to training folder.')
    parser.add_argument('-gt','--ground_truth', required=True, help='Path to ground-truth folder.')
    try:
        args = parser.parse_args()
        epochs_id = int(args.n)
        path_train = args.train
        path_gt = args.ground_truth
    except:
        parser.print_help()
        sys.exit(1)

    # Create for weight folder
    if not os.path.exists(path_model):
        os.makedirs(path_model)

    # Load existing model
    if epochs_id == 0:
        model_qn = get_q_network("0")
    else:
        model_qn = get_q_network(path_model + '/model_epoch_' + str(epochs_id) + '_h5')

    #### LOAD MODEL ####
    model_vgg = obtain_compiled_vgg_16(path_vgg)

    #### LOAD GROUND-TRUTH DATAFRAME ####
    df = pd.read_csv(path_gt, sep=" ", header = None)
    df.columns = ["ID", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]
    # Delete NA and remove 'lost' and 'occluded' rows
    df.dropna; clean_df = df.drop(df[(df.lost == 1) | (df.occluded == 1)].index)
    min_ID = min(df.ID) ; max_ID = max(df.ID)

    ######## LOOP THROUGH EPOCHS #########
    for i in range(epochs_id, epochs_id + epoch_size):      # Loop through Epochs
        print('epoch %d ...' % i)
        for j in range(min_ID,max_ID+1):                        # Loop through Cars
            print('\tcar %d ... ' % j)
            car = df.iloc[ clean_df[clean_df.ID == j].index ]
            total_frames = car.shape[0]

            for k in range(0,frames_step):
                # Initial Frame & Annotation info
                annotation = get_annotation(car, k)
                last_annotation = np.zeros([5], dtype=np.int)

                # Loop through Frames for a Car
                for l in range(frames_step+k,total_frames,frames_step):
                    last_annotation = annotation

                    # Extract annotation info
                    annotation = get_annotation(car, l)

                    # Get frame
                    image_path = path_train + '/' + str(annotation[0]) + '.jpg'
                    print('\t\tframe %s ... ' % str(annotation[0]) )
                    img = np.array( image.load_img(image_path, False) )

                    original_shape = ( img.shape[0] , img.shape[1] )

                    # Get Ground Truth mask
                    gt_mask = np.zeros([int(img.shape[0]), int(img.shape[1])])
                    gt_mask[int(annotation[3]):int(annotation[4]), int(annotation[1]):int(annotation[2])] = 1

                    # Get Region mask
                    region_mask = np.zeros( original_shape )
                    region_mask[last_annotation[3]:last_annotation[4], last_annotation[1]:last_annotation[2]] = 1

                    # Get Region image
                    region_img = img[last_annotation[3]:last_annotation[4], last_annotation[1]:last_annotation[2]]

                    # initialize history vector
                    # NOTE: 36 = (9 actions)*(4 )
                    history_vector = np.zeros([36])

                    # get initial state
                    state = get_state(region_img, history_vector)

                    # initialize all flags
                    step = 0
                    status = 1
                    action = 0
                    reward = 0

                    # initial bounding box's info
                    xmin = last_annotation[1]
                    xmax = last_annotation[2]
                    ymin = last_annotation[3]
                    ymax = last_annotation[4]

                    # get image info
                    img_height, img_width = original_shape

                    # get bounding box info
                    bb_width = int(last_annotation[2]) - int(last_annotation[1])
                    bb_height = int(last_annotation[4]) - int(last_annotation[3])
                    bb_shape = (bb_height,bb_width)

                    # Get the first mask
                    (mask, _, _, _, _) = shift_mask(9, xmin, xmax, ymin, ymax, original_shape, bb_shape)
                    # Take number_of_steps to reach the gt mask
                    while (status == 1) and (step < number_of_steps):
                        sys.stdout.write('.')
                        sys.stdout.flush()

                        # Experience Replay Storage
                        # Fill up the experience replay buffer before updating the Q-network
                        if len(replay) < buffer_experience_replay:
                            y = get_target_y(action, xmin, xmax, ymin, ymax, bb_shape, gt_mask, mask, img, model_qn, history_vector)
                            replay.append((state,y))
                        # Update experience replay buffer and Q-network
                        else:
                            if p < buffer_experience_replay-1:
                                p += 1
                            else:
                                p = 0
                            y = get_target_y(action, xmin, xmax, ymin, ymax, bb_shape, gt_mask, mask, img, model_qn, history_vector)
                            replay[int(p)] = (state,y)
                            minibatch = random.sample(replay, batch_size)
                            X_train, Y_train = zip(*minibatch)
                            X_train = np.array(X_train)
                            Y_train = np.array(Y_train)
                            X_train = X_train.astype("float32")
                            Y_train = Y_train.astype("float32")
                            X_train = np.squeeze(X_train, axis=2)

                            model_qn.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1, verbose=1)

                        # Make an action
                        qval = model_qn.predict(state.T, batch_size=1)
                        step += 1
                        # epsilon-greedy policy
                        if random.random() < epsilon:
                            # Take a random action
                            action = np.random.randint(1, 9)    # Choose action 1-8 only
                        else:
                            # Take the optimal action
                            action = (np.argmax(qval))+1

                        # Take the action and saved the new state
                        (region_mask, xmin, xmax, ymin, ymax) = shift_mask(action, xmin, xmax, ymin, ymax, original_shape, bb_shape)
                        region_img = img[ymin:ymax,xmin:xmax]
                        mask = region_mask
                        history_vector = update_history_vector(history_vector, action)
                        new_state = get_state(region_img, history_vector)
                        state = new_state
                        if action == 9:     # Terminal action
                            status = 0      # Break out of action loop
        # Decrement Epsilon (From Exploration to Exploitation)
        if epsilon > 0.1:
            epsilon -= 0.05
            epsilon = max(epsilon,0)
            print("========== Epsilon - {} ============".format(epsilon))

        # Save model's weights after every epoch
        model_save_path = path_model + '/model_epoch_{}_e{:03}_h5'.format(str(i),int(round(epsilon,2)*100))
        model_qn.save_weights(model_save_path, overwrite=True)
