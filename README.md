# Birdify

Aiming to be a better bird classification tool

## Introduction

### What is the problem

We want to classify bird species based on bird images. We tested many different configurations that can affect the test accuracy to generate better predictions. The configurations include:

- Different models, such as the resnet18, resnet50, resnet101, inceptionv4, efficientNetV1
- Hyper-parameter settings such as the learning rate and the stepwise learning rate decay
- Using ImageNet pre-trained weights or none 
- Image resolutions
- Cross validation: enabled or none
- Dropout in the FC layer: none, or p = 0.5

### Dataset

We use the [dataset](!https://www.kaggle.com/c/birds21sp/data) provided on Kaggle.

The training set contains 555 species of birds, each with around 70 images. In total, the training set contains 38562 images.

The test set contains 10000 entries. Our current test accuracy is based on 10% of it.

## Approach

### What techniques did you use?

We first used the training method with SGD optimizer, Cross Entropy Loss, and hyper-parameters including epochs=5, learning_rate=0.01, momentum=0.9, decay=0.0005. We tried to increase the training epochs to see where the test accuracy stops to increase. After we choose a reasonable number of epoch, we increased the resolution of input images.

Next, we tried different models using the same hyperparameter setting to see which ones outstand. We picked Inception V4 and Efficient Net.

In the end, we added cross validation to see how it affects the performance of these two model we picked.

### - What problems did you run into?

- Hard-ware limit / Training time limit

  Number of epochs:

  5 -> 35: Loss of Inception V4 decrease to as low as 0.02. 30 mins per epoch.

  -> 20: Loss of Efficient Net decrease to as low as 0.006. 60 mins per epoch.

  Resolution of input image: 128 -> 214 -> 300 -> 512 -> 600

- Overfitting

  Training accuracy becomes 100% from the 12th epoch.

  Using different learning rate during different epochs.

- Pre-trained or Not

- Hard to tell if it overfits from the accuracy

### - Why did you think this approach was better than other options?

- Using 10-fold cross validation, elaborate loss, accuracy, test accuracy difference

## Experiments

### - Try multiple things to see what works better

- Different models on the same basic hyper-parameter setting: ResNet18, ResNet50, ResNet101, Inception V3, Inception V4, Efficient Net V1-B5.

- Diagrams

## Results

### - what worked better

Cross Validation reduced loss.

### - Maybe have some nice charts and graphs here

## Discussion

### - What worked well and didn’t and WHY do you think that’s the case?

### - Did you learn anything?

### - Can anything in this project apply more broadly to other projects?
