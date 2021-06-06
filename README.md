# Birdify

Aiming to be a better bird classification tool

## Introduction

### - What is the problem

We want to classify bird species based on bird images. We want to test how different models,hyperparameter settings, image resolutions, and cross validation affect the test accuracy.

### - Dataset

We use the [dataset](!https://www.kaggle.com/c/birds21sp/data) provide on Kaggle.

The training set contains 555 species of birds, each with around 70 images. In total, the training set contains 38562 images.

The test set contains 10000 entries. Our current test accuracy is based on 10% of it.

## Approach

### - What techniques did you use?

We first used the training method with SGD optimizer, Cross Entropy Loss, and hyper-parameters including epochs=5, learning_rate=0.01, momentum=0.9, decay=0.0005. We tried to increase the training epochs to see where the test accuracy stops to increase. After we choose a reasonable number of epoch, we increased the resolution of input images.

Next, we tried different models using the same hyperparameter setting to see which ones outstand. We picked Inception V4 and Efficient Net.

In the end, we added cross validation to see how it affects the performance of these two model we picked.

### - What problems did you run into?

- Hard-ware limit / Training time limit

  Number of epochs: 35

  Resolution of input image:

- Overfitting

  Training accuracy becomes 100% from the 12th epoch.

  Using different learning rate during different epochs.

-

### - Why did you think this approach was better than other options?

- Using 5-fold cross validation, the training accuracy doesn't increase to 100% that fast. Even if it's high during training, it falls back during validation.

## Experiments

### - Try multiple things to see what works better

Diagrams

## Results

### - Or maybe here is where you talk about what worked better idk

### - Maybe have some nice charts and graphs here

## Discussion

### - What worked well and didn’t and WHY do you think that’s the case?

### - Did you learn anything?

### - Can anything in this project apply more broadly to other projects?
