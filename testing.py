# Usage:
#       python testing.py -w=./model_qn_weights/model_epoch_19_e010_h5 -t=./test_images/ -gt=./test_images/PS12_1_7_gt.txt
#
#   OR
#       python testing.py
#
import os, sys, random, argparse
import numpy as np
import pandas as pd
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import backend as K
K.set_image_dim_ordering('tf')
K.image_dim_ordering()
from keras.preprocessing import image

from reinforcement import *
from features import *
from image_helper import *

# Make inference result reproducible
random.seed(1234)

if __name__ == "__main__":

    #### PARAMETERS ####
    number_of_steps = 20
    frames_step = 30
    #### PATHS ####
    path_vgg = "./vgg16_weights_tf_dim_ordering_tf_kernels.h5"

    # Parse all script arguments
    parser = argparse.ArgumentParser(description='This python script test on the image frames in the testing folder and annotations in the ground-truth folder')
    parser.add_argument('-w','--weight',default="./model_qn_weights/model_epoch_19_e010_h5",help='Path to the trained model weights')
    parser.add_argument('-t','--test', default="./test_images", help='Path to test folder.')
    parser.add_argument('-gt','--ground_truth', default="./test_images/PS12_1_7_gt.txt", help='Path to ground-truth folder.')
    try:
        args = parser.parse_args()
        path_weights = args.weight
        path_test = args.test
        path_gt = args.ground_truth
    except:
        parser.print_help()
        sys.exit(1)

    try:
        #### MODELS ####
        model_vgg = obtain_compiled_vgg_16(path_vgg)
        # Load existing model
        model_qn = get_q_network(path_weights)
    except:
        print("Please verify vgg and model weight path.")
        sys.exit(1)

    #### LOAD GROUND-TRUTH DATAFRAME ####
    df = pd.read_csv(path_gt, sep=" ", header = None)
    df.columns = ["ID", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]
    # Delete NA and remove 'lost' and 'occluded' rows
    df.dropna;
    clean_df = df.drop(df[(df.lost == 1) | (df.occluded == 1)].index)
    min_ID = min(df.ID) ; max_ID = max(df.ID)

    # Loop through the Cars in the Video
    for j in range(min_ID,max_ID+1):
        print('\tcar %d ... ' % j)
        car = df.iloc[ clean_df[clean_df.ID == j].index ]
        total_frames = car.shape[0]

        # Get random frame from the car
        k = random.randint(0,total_frames-frames_step-1)
        # Initial Frame & Annotation info
        annotation = get_annotation(car, k)
        last_annotation = np.zeros([4], dtype=np.int)
        last_annotation = annotation

        # Extract annotation info
        l = k + frames_step
        annotation = get_annotation(car, l)

        # Get frame
        string = path_test + '/' + str(annotation[0]) + '.jpg'
        print('\t\tframe %s ... ' % str(annotation[0]) )
        img = image.load_img(string, False)
        img = np.array(img)
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
        # NOTE: 20 = (5 actions)*(4 )
        history_vector = np.zeros([36])
        # get initial state
        state = get_state(region_img, history_vector)

        # initialize all flags
        step = 0
        status = 1
        action = 0

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
        bb_shape = (bb_height, bb_width)

        orig_img = region_img
        arr_regionImg = []
        actions = []

        # Inference and perform actions to target for a fixed number of steps
        while (status == 1) & (step < number_of_steps):
            step += 1

            # Have the model predict the best action
            qval = model_qn.predict(state.T, batch_size=1)
            action = (np.argmax(qval))+1
            if action == 9:             # terminal action
                status = 0
            else:
                region_mask, xmin, xmax, ymin, ymax = shift_mask(action, xmin, xmax, ymin, ymax, original_shape, bb_shape)
                region_img = img[ymin:ymax,xmin:xmax]

                # Used to print output
                arr_regionImg.append(region_img)
                actions.append(action)

            # Make new state the current state for the next pass
            history_vector = update_history_vector(history_vector, action)
            new_state = get_state(region_img, history_vector)
            state = new_state

        # Print the outputs of each inference
        print_outputs_9(orig_img, arr_regionImg,actions, j)
