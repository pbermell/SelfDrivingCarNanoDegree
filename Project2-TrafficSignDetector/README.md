# Trafic sign recognition
Self Driving Car Engineer, Udacity NanoDegree. Project 2
April 2017

[//]: # (Image References)
[image01]: ./rimages/ill01.png
[image02]: ./rimages/ill02.png
[image03]: ./rimages/ill03.png
[table1]: ./rimages/t01.png
[image04]: ./rimages/ill04.png
[image04]: ./rimages/ill05.png
[image06]: ./rimages/ill06.png
[image07]: ./rimages/ill07.png
[table2]: ./rimages/t02.png
[table3]: ./rimages/t03.png
[image08]: ./rimages/ill08.png
[image09]: ./rimages/ill09.png
[image10]: ./rimages/ill10.png
[image11]: ./rimages/ill11.png


## 0. Introduction
This document reports on the work carried out to implement a traffic sign recognition system using the learning gained in the Udacity Self Driving Car Nanodegree course. Project 2 focuses on the implementation of Convolutional Neural Networks using TensorFlow. This is my first attempt to implement a Neural Network and it has been an intensive exercise of learning. The report is part of the Project 2 deliverables and documents this journey. The implementation of this project was in great deal adapted from the one created by Jeremy Shannon, a fellow student on the course [1]. Any re-use of code from others has been analysed and adapted. I was particularly interested in understanding general python syntax and openCV syntax and Jeremy’s notebook is a great source of knowledge for this. 

### 0.1 Implementation platform
The project was implemented on an Acer Aspire v15 Nitro – Black Edition Laptop, (Intel Core i7-4720HQ processor, 16Gb RAM), equipped with an NVIDIA GEFORCE GTX 960m GPU. The  software implementation uses Docker version 17.03.1-ce hosted on Ubuntu 16.04 LTS. On top of this, the solution uses the NVIDIA-DOCKER software solution enabling the containers to access the GPU of the host computer. The project uses a modified version of the official Python 3 and GPU enabled Tensorflow Docker image (“tensorflow/tensorflow:latest-gpu-py3”), available at the Docker Hub. While the default base image supports GPU and python 3, aditional layers were built into the container in order to implement the OpenCV 3 dependency for the project. With this setup, the training times for the convolutional neural network (Sermanet architecture), using the GPU were less than 5 minutes.

## 1. Dataset summary & exploration
The dataset is a compilation of traffic signs made available by a German institution called: “Institut fur Neuroinformatik”. The first approach to understand the dataset is querying it so we can undertsand the size and the shape of its images content. The following elements are queried from the dataset using python commands:  

Number of training examples = 34799 ; Number of validation examples = 4410 ; Number of testing examples = 12630 ; Image data shape = (32, 32, 3) ; Number of classes = 43

### 1.1 Exploratory visualisation
While these figures are fundamental to understand the dataset, further confidence was built through several visualisations. First of all a sample of the 32x32x3 images of the dataset was extracted (Figure1). 

![alt text][image01]

Figure1: a sample of the images in the dataset

The next question about the dataset concern the distribution of the training set samples among the 43 labeled classes. Fot that, a distribution histogram was plotted and presented in the Jupyter notebook, (Figure2). 

![alt text][image02]

Figure2: distribution of classes in the training set

The images on the dataset are not evenly distributed across the classes provided by the labelled data as it can be appreciated in the plot. A query on the data showed that the least populated class holds 180 images while the most populated one contains 2010 image samples. This clearly builds the case for artificially creating data samples for the less populated clasess. This is a recommended practice on the notion that training a neural network over non uniform distribution of labelled data can biase the predictions. 

![alt text][image03]

Figure3: one image of each class of the dataset.

An intuitive view to the data completes the exploratory visualisation process. For instance, figure 3 shows an example picture of each class from the dataset. Exploring this visualisation over multiple images of the 43 classes may look irrelevant. However, it gives an idea of the sort of traffic signs images the dataset is built upon. 

## 2. Design and test a model architecture
### 2.1 Data preprocessing

Basic preprocessing of the dataset was achieved by first re-shuffling randomly the dataset which was found to be arranged by classes. In Table 2, the class labels of the first 500 image samples are displayed before (left) and after (right) the re-shuffling.

![alt text][table1]

Table 1: 500 first labels on the dataset, before and after re-shuffling

#### 2.1.1 Grayscale conversion

Further pre-procesing options for this dataset confront various choices. The first one is whether if images should be converted to grayscale or not. In this project, the choice was made to convert images to grayscale. Intutively, the conversion tends to increase the contrast on the images as it can be seen in the multiple plot, (figure 4). 

![alt text][image04]

Figure 4: grayscale vs. color images from the dataset

Besides this intuition, it was assumed that the grayscale conversion would speed up the computation speed and therefore facilitate faster iterations in the coding and exploration of the solution. 

#### 2.1.2 Normalisation

It is recommended in various forums to reduce as much as possible the variation between the values of the features used for learning. However, I dont really understand in details the reasons behind. That leads to standardisation and normalisation as common practices to frame the 0 to 255 range of the pixels intensity values into a smaller range solution. 2 approaches to normalisation were tested in the notebook. I finally choose the solution that makes a distribution around the center value (128). Further support for this choice can be found in the studies ran by Jessica Yung, another fellow student, in her project [2]. Intutively, the normalized and original images do not differ much as it can be seen in the 128 normalised vs only grayscaled comparison, (figure 5). I leave the extraction of conclusions out of this to future exercises. 

![alt text][image05]

Figure 5: normalised vs. original grayscaled images

#### 2.1.3 Image transformations

Although the last preprocessing operation was perceived as an optional feature of the assignment, I decided to explore the solution from Jeremy [1]. I implemented only the translation and scaling transformations to generate additional data and therefore manage the risk of biasing the dataset due to the non-uniform distribution of the classes mentioned above. This practice is refered as “jittering” in various information sources. This part of the project lead me to explore and play with OpenCV. For each transformation, the values of the translation and scaling factor are chosen randomly within a given range, (-4 to 4 pixels). The result of this transformation is a more uniform distribution of the training samples across the labeled classes. The code generates new transformed images in the class bins containing less than 800 images. The transformations are applied subsequently to images existing in each bin. 

#### 2.1.4 Regenerate the validation dataset

I followed the approach on [1] to generate a validation set as as subset of the augmented dataset generated through OpenCV transformations. This appealed as a valid notion since the validation set comes from a more uniformally distributed sample across the 43 classes. I also learned from one of my collegues at work about the use of the “train_split_function” and how useful is the parameter  “random_state” to make controlled random samples, (so they are random but in the same way!). The result is a dataset whose distribution has at least 800 images for each of the 43 classes, (figure 6).

![alt text][image06]

Figure 6: distribution in the jittered dataset

### 2.2 Model architecture

The choice of implementation for my first neural network is the Sermanet architecture reported in [3]. This network design is a bit more sophisticated than LeNet [4], used in the coursework. An additional argument to use it is its application in traffic signs classification.

![alt text][image07]

Figure 7: Sermanet network

A key aspect to implement this network is in the choice of inputs and outputs for each of the layers. The architecture of Sermanet is very similar to the LeNet. The general features of the network are described in Table 2. Being my first neural network, my focus has been on understanding well an architecture rather than to explore new ones. 

![alt text][table2]

Table 2: Network architecture

### 2.3 Model training

For the model training, I proposed an initial set of paramteres partly based on the exploration in [1]. A small sensitivity analysis on the influence of performance and learning time due to key parameters has been carried out to explore improvements. The summary of this described in table 3. Not all the parameters are investigated. However, I considered these important ones on my first approach to a Convolutional Neural Network. 

![alt text][table3]

Table 3: Sensitivity analysis of key parameters on the training performance

Looking at the starting point of this sensitivity analysis, the most significant change is is the reduction of Batch sizes from 128 to 100 and a small modification on the training rate. Therefore, it is fair to say that the model was well tuned from the beginning and I just have investigated the impact of changes. 

For the training optimisation, I used the AdamOptimiser. The choice was based on the recommendation made in the course lectures. 

### 2.4 Solution approach

The solution uses the separated test dataset to evaluate the network trained in previous sections. The performance obtained is 92.4%. 

## 3. Test a model on new images

### 3.1 Acquiring new images

For the testing of the resulting neural network I downloaded 5 images after a Google Images search using the keywords: “German traffic signs”. The images were cropped and scaled to a 32x32 pixels size, (Illustration 8). 

![alt text][image08]

Figure 8: Five test images

### 3.2 Performance on new images

The implementation predicts 100% of the new images. It is no surprise since they are very similar to those in the training set. In a previous test I run the model over a more challenging dataset and the performance was clearly lower. An example image of this dataset is shown in figure 9.

![alt text][image09]

Figure 9: previously tested images (non german, giving poor classification results)

Figure 10 depicts the results of the classification using the 5 German traffic signs dataset and the performance of the classification based on images of the training set. 

![alt text][image10]

Figure 10: classification results

The performance obtained in the test was above 90% and it is high in the new images. However, the new images are very similar to the ones in the dataset. The results were significantly lower on the previous test where images are not from German traffic signs. 

### 3.3 Model certainty – softmax probabilities

The softmax probability analysis shown in figure 11 also confirms the performance of the model. 

![alt text][image11]

Figure 11: Softmax results

## 4. Concluding remarks

In this project, I have built the inderstanding required to implement a convolutional neural network to classify images. This is the first immplementation of this type of system I have ver done. It has been quite a journey and a very steep learning curve given my limited experience in Python programming. I am in debt to the implementations in [1] and [2] from which I have learn a lot. I haven’t reused any part of their code without analysing it. The result is probably a longer development time than what other students commited to the project. However, I have enjoyed the learning and the experience. Despite of some uncertanties, I have a clear picture of the architecture of a Neural Network and the key parameters influencing its performance. I consider this work quite an achievement done on top of my day to day work at Airbus CTO office. I have alredy a couple of ideas of what I could do with the techniques learnt here for instancce in the area of manufacturing engineering digitalisation. 

### 4.1 Future work related to this project

Resulting form this experience my next steps in the further understanding Neural Networks include:

..* Re-visiting the lectures to consolidate some of the concepts. I am still a bit unclear about some aspects such as cross-entropy and softmax probabilities.
..* Keep practising python programming to get faster on further assignments.
..* Understanding the optional part of the assignment. 

## References

[1] Jeremy Shannon’s submission to Project 2. https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb

[2] Jessica Yung’s studies on normalisation at her Project 2 submission.  https://github.com/jessicayung/self-driving-car-nd/blob/master/p2-traffic-signs/Comparison%20of%20model%20performance%20using%20original%2C%20standardised%20and%20normalised%20data.ipynb

[3] Sermanet, P., & LeCun, Y. (2011, July). Traffic sign recognition with multi-scale convolutional networks. In Neural Networks (IJCNN), The 2011 International Joint Conference on (pp. 2809-2813). IEEE.

[4] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

