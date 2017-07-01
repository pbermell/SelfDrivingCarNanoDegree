# Trafic sign recognition
Self Driving Car Udacity NanoDegree. Project 2
April 2017

[//]: # (Image References)
[image01]: ./rimages/ill01.png



## 0. Introduction
This document reports on the work carried out to implement a traffic sign recognition system using the learning gained in the Udacity Self Driving Car Nanodegree course. Project 2 focuses on the implementation of Convolutional Neural Networks using TensorFlow. This is my first attempt to implement a Neural Network and it has been an intensive exercise of learning. The report is part of the Project 2 deliverables and documents this journey. The implementation of this project was in great deal adapted from the one created by Jeremy Shannon, a fellow student on the course [1]. Any re-use of code from others has been analysed and adapted. I was particularly interested in understanding general python syntax and openCV syntax and Jeremy’s notebook is a great source of knowledge for this. 
### 0.1 Implemenation platform
The project was implemented on an Acer Aspire v15 Nitro – Black Edition Laptop, (Intel Core i7-4720HQ processor, 16Gb RAM), equipped with an NVIDIA GEFORCE GTX 960m GPU. The  software implementation uses Docker version 17.03.1-ce hosted on Ubuntu 16.04 LTS. On top of this, the solution uses the NVIDIA-DOCKER software solution enabling the containers to access the GPU of the host computer. The project uses a modified version of the official Python 3 and GPU enabled Tensorflow Docker image (“tensorflow/tensorflow:latest-gpu-py3”), available at the Docker Hub. While the default base image supports GPU and python 3, aditional layers were built into the container in order to implement the OpenCV 3 dependency for the project. With this setup, the training times for the convolutional neural network (Sermanet architecture), using the GPU were less than 5 minutes.
## 1. Dataset summary & exploration
The dataset is a compilation of traffic signs made available by a German institution called: “Institut fur Neuroinformatik”. The first approach to understand the dataset is querying it so we can undertsand the size and the shape of its images content. The following elements are queried from the dataset using python commands:  

Number of training examples = 34799 ; Number of validation examples = 4410 ; Number of testing examples = 12630 ; Image data shape = (32, 32, 3) ; Number of classes = 43

### 1.1 Exploratory visualisation
While these figures are fundamental to understand the dataset, further confidence was built through several visualisations. First of all a sample of the 32x32x3 images of the dataset was extracted (Illustration1). 

![alt text][image01]

Figure1: a sample of the images in the dataset


