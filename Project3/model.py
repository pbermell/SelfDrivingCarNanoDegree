#################
# INITIALISATION

import csv
import cv2
import numpy as np
import os
#import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#%matplotlib inline


# PROGRAM SETTINGS
##################################
### File management
# Location of driving log csv file
csv_path = './data/driving_log.csv'
img_directory = './data/'

# this is are the contents of the first line of a driving_log.csv file:
# Note the root directory is "data" while the files are in the IMG directory
# under the root directory

# IMG/center_2016_12_01_13_30_48_287.jpg
# IMG/left_2016_12_01_13_30_48_287.jpg
# IMG/right_2016_12_01_13_30_48_287.jpg
# 0
# 0
# 0
# 22.14829

##################################
##################################
##########################
### CNN Key parameters ###
learn_indicator = 'mse'
learning_rate = 1e-4
batch_size = 128
n_epochs = 20


##########################


#######################
## SUPPORTING FUNCTIONS
# 2 Supporting Functions included in the program
# preprocess_img: transforming single images
# preparing_training_data: Is the generator to optimise memory used for training


def preprocess_img(img):
    '''
    This function is called at the preparation of the training set 
    Input is an 160x320x3 image 
    Output is an image with various transformations.

    Same transformations are applied on drive.py the difference being:
    Here at model.py: input image is BGR (due to OpenCV reader object image output).
    Output image is in YUV color space as recommended by NVIDIA. 

    '''

    # original shape of images is 160x320x3
    # input shape for Nvidia NN is 66x200x3
    # crop original image to remove sky (50 pixels from top) and car front (20 pixels from bottom)
    new_img = img[50:140, :, :]
    # Apply a Gaussian Blur Filter
    new_img = cv2.GaussianBlur(new_img, (3, 3), 0)
    # resize the image to 66x200x3
    new_img = cv2.resize(new_img, (200, 66), interpolation=cv2.INTER_AREA)
    # From BGR to YUV
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img


def prepare_training_data(images, angles, batch_size=batch_size):
    '''
     This is the generator function
    '''
    images, angles = shuffle(images, angles)
    X, y = ([], [])
    while True:
        for i in range(len(angles)):
            img = cv2.imread(img_directory + images[i])
            img = preprocess_img(img)
            angle = angles[i]
            X.append(img)
            y.append(angle)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([], [])
                # print(X,y)
                images, angles = shuffle(images, angles)
            # flip images
            if abs(angle) > 0.33:
                img = cv2.flip(img, 1)
                angle *= -1
                X.append(img)
                y.append(angle)
                if len(X) == batch_size:
                    yield (np.array(X), np.array(y))
                    X, y = ([], [])
                    # print(X,y)
                    images, angles = shuffle(images, angles)


#####################
### MAIN PROGRAM
#####################


# Read csv file

with open(csv_path, newline='') as f:
    driving_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))

# print(len(driving_data))
#     print(driving_data[1])



# extract content of each row to 2 arrays

images = []
angles = []
for row in driving_data[1:]:
    # remove 0 speed lines
    if float(row[6]) < 0.1:
        continue
    # get centre image path and angle
    images.append(row[0])
    angles.append(float(row[3]))
    angles_count = len(angles)
    # get left image and angle
    images.append(row[1])
    angles.append(float(row[3]) + 0.25)
    # get right image and angle
    images.append(row[2])
    angles.append(float(row[3]) - 0.25)

image_paths = np.array(images)
print(len(images))
angles = np.array(angles)
print(len(angles))

# compute the data for an angles histogram distribution with n bins
# and the anerage amount of samples in each bin

n_bins = 33
avg_samples_per_bin = len(angles) / n_bins
# print(avg_samples_per_bin)
hist, bins = np.histogram(angles, n_bins)
# print(hist)
# print(bins)


## Use this code on a Jupyter notebook to visualise the distribution of angles
## in the dataset before pre-processing
# hist, bins = np.histogram(angles, n_bins)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.plot((np.min(angles), np.max(angles)))
# plt.show()


# Based on the deviation from the average samples per bin
# compute a multiplication factor to remove samples from the
# ditribution for each bin.
# i.e. 1.0 won't remove samples, 0.8 will keep 80% of the samples
#

keep_probs = []
target = avg_samples_per_bin * .5
for i in range(n_bins):
    if hist[i] < target:
        keep_probs.append(1.)
    else:
        keep_probs.append(1. / (hist[i] / target))

print(keep_probs)
print(len(keep_probs))

# compute a list of samples to be removed in each bin
# based on keep_probs array

remove_list = []
for i in range(len(angles)):
    for j in range(n_bins):
        if angles[i] > bins[j] and angles[i] <= bins[j + 1]:
            # if angles[i] > keep_probs[j]:
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)

print(len(remove_list))
# print(remove_list[100:120])

# delete the samples in images and angles specified
# by the remove_list

images = np.delete(images, remove_list, axis=0)
angles = np.delete(angles, remove_list)
print(len(images))
print(len(angles))
# print(images[6000:6010])
# print(angles[6000:6010])


# img = cv2.cvtColor(images, cv2.COLOR_YUV2BGR)
# img = cv2.imread(images[1])
# print(img)
# cv2.imshow('image', img)


## Use on a Jupyter notebook to visualise the distribution of angles
## in the dataset after pre-processing
# hist, bins = np.histogram(angles, n_bins)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.plot((np.min(angles), np.max(angles)))
# plt.show()


# use train_test_split to generate the training set from the images and angles arrays
#

images_train, images_test, angles_train, angles_test = train_test_split(images, angles,
                                                                        test_size=0.05,
                                                                        random_state=42)


# print(len(images_train))
# print(len(angles_train))


#####################################
# CNN HERE
#####################################
# NVIDIA CNN
# Based on:
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

model = Sequential()
# Normalisation
model.add(Lambda(lambda x: x /127.5 - 1.5, input_shape=(66,200,3)))

# three 5x5 Convolutional layers (output depths 24, 36, 48 and 2x2 strides)
model.add(Convolution2D(24,5,5, subsample=(2,2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(36,5,5, subsample=(2,2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(48,5,5, subsample=(2,2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

# two 3x3 Convolutional layers (output depths 64 and 64)
model.add(Convolution2D(64,3,3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(64,3,3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

# Flatten layer
model.add(Flatten())

# three fully connected layers (depths 100, 50, 10)
model.add(Dense(100, kernel_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dense(50, kernel_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dense(10, kernel_regularizer=l2(0.001)))
model.add(ELU())

# A fully connected output layer
model.add(Dense(1))

# Compile and train
model.compile(optimizer=Adam(lr=learning_rate), loss= learn_indicator)


######################################


train_prep = prepare_training_data(images_train, angles_train, batch_size=batch_size)
val_prep = prepare_training_data(images_train, angles_train, batch_size=batch_size)

#model.fit(train_prep[0], train_prep[1], validation_split=0.2, shuffle=True, nb_epoch=3)

# Saves a file on each epoch and details the val_loss in the file name
checkpoint = ModelCheckpoint('model{epoch:02d}-{val_loss:.2f}.h5')



model.fit_generator(train_prep,
                    validation_data=val_prep,
                    epochs= n_epochs,
                    validation_steps=(len(images_test)/batch_size),
                    steps_per_epoch=(len(images_train)/batch_size),
                    verbose = 1,
                    callbacks = [checkpoint]
                    )

# visualise a summary of the CNN
print(model.summary())


model.save_weights('./model.h5')
json_string = model.to_json()
with open('./model.json', 'w') as f:
    f.write(json_string)