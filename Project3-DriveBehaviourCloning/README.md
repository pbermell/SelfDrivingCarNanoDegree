# Behavioural Cloning
Self Driving Car Engineer, Udacity NanoDegree. Project 3, May 2017.


[//]: # (Image References)
[image01]: ./ReadmeImages/figure01.png


## 0. Introduction

This document reports on the work carried out to implement a behavioural cloning system from the learning gained in the Udacity’s Self Driving Car Nanodegree course. The project focuses on the implementation of deep learning and computer vision techniques enabling the autonomous drive of a car in a simulated environment. The key learning outcome of this project is the ability to exploit a trained convolutional neural network into a real time computer vision application to drive the car using the resulting model. The other main take from this project is the importance of the data used to train a model. Finally, the project is a great way of consolidating many of the concepts we have studied so far in the course. This report captures key insights gained on the project. Execution required extensive inputs from the Udacity community since it had some uncertainties. Remarkably these are: the fact that the loss does not correlate very well with the performance of the driving; and the difficulty to capture driving data without an analog joystick as pointed out in the forums. As in my previous project, I used Jeremy Shannon’s implementation as an starting point for my own, [1]. Jeremy’s implementation includes various OpenCV transformations and jitterings to the input images. However, it recognises the importance of the data to produce satisfactory results. Based on this, the working hypothesis is that I could train the model with the minimum amount of data and image transformations. This hypothesis is successfully validated in the project which uses only the dataset provided by Udacity and no synthetic image image genration, (a process often called "jittering").

### 0.1 Implementation platform

The project was implemented on an Acer Aspire v15 Nitro – Black Edition Laptop, (Intel Core i7-4720HQ processor, 16Gb RAM), equipped with an NVIDIA GEFORCE GTX 960m GPU. The  software implementation for learning uses Docker version 17.03.1-ce hosted on Ubuntu 16.04 LTS. On top of this, the solution uses the NVIDIA-DOCKER software solution enabling the containers to access the GPU of the host computer. The container used for training in the project  is a modified version of the official Python 3 and GPU enabled Tensorflow Docker image (“tensorflow/tensorflow:latest-gpu-py3”), available at the Docker Hub. While the default base image supports GPU and python 3, additional layers were built into the container in order to implement the OpenCV 3 dependency for the project. For driving, the project uses the official Udacity Docker image with the exception of an update of Keras to version 2.0.3 which is the same used by the training container. 

## 1. Required files and quality of code

All required files are uploaded to the Udacity project repository for review and evaluation. The code is annotated to help on its interpretation. Most of the engineering process of writing and testing the model.py code has been carried out in a Jupyter notebook decomposed in cells structured according to the main functionalities of the program as described below.

### 1.1 model.py

As expected, the code uses a generator to prevent excessive usage of computer memory. Testing without the generator rendered unusable the development laptop using in the project. Key parts of the code are:

**Initialisation:** imports the libraries and components necessary to perform the training.

**Program settings:** collects the variables and settings related to the file management (images and csv files), and the key settings of the neural network. As it is setup, the `model.py` file has to be in a directory containing another directory called “data”. The latter contains the csv file provided by Udacity and the IMG directory containing all the images of the dataset. 

**Supporting functions:** the preprocess_img function is used to transform any single images coming from the simulator, (shape 160x320x3 RGB color schema), so they can be utilised as input to the neural network, (shape 66x200x3 YUV color schema).  The function `prepare_traing_data` is the generator used in the project to “yield” images to the neural network in batches defined in the parameter `batch_size` in order to manage efficiently the memory used by the system. 

**Main program.** This part of the code executes the pipeline transforming the images and labels in the computer file system into a preprocessed dataset ready to be fed into the neural network. First, the program opens the csv file and loads the data frame into a list, (`driving_data`). An iteration through this list removes rows of the data frame in which the captured speed is near zero. As recommended by [1] and others on the Udacity forum, this iteration includes adjustments on the angle reading of left and right hand images by +0.25 and -0.25.  The result is the two arrays with the same amount of elements: `images[]`, `angles[]`. 

The second element of the main program is the pre-processing of these arrays in order to analyse and improve the distribution of angles in the dataset. The recording of steering angles in driving activity usually shows a high frequency distribution of 0 degree angles of the steering wheel. As recommended in [2] the adjustment of this distribution on a training dataset is important to remove the bias towards driving straight in the end result. The resampling of the dataset uses the approach from [1]. In Illustration 1 (a) and (c), the distribution of angles captured in the dataset provided by Udacity and in an enriched dataset created manually using the simulator. 

![alt text][image01]
