# Birdify

Aiming to be a better bird classification tool

## Introduction

### What is the problem

We want to classify bird species based on bird images. We tested many different configurations that can affect the test accuracy to generate better predictions. 

### Dataset

We use the [dataset](!https://www.kaggle.com/c/birds21sp/data) provided on Kaggle.

The training set contains 555 species of birds, each with around 70 images. In total, the training set contains 38562 images.

The test set contains 10000 entries. Our current test accuracy is based on 10% of it.

### Team

Bowen Xu

Wenqing Lan

## Approach

### What techniques did you use?

We first started the training with the pretrained `resnet18` model. We also used the SGD optimizer and Cross Entropy Loss. (We compared the popular optimizers such as Adam, AdamW, and SGD. [This article](https://towardsdatascience.com/why-adamw-matters-736223f31b5d) shows that the SGD is still the optimizer that produces more generalizable models. Therefore we chose to use the SGD optimizer throughout the project.) Our initial hyper-parameters included epochs = 5, learning rate = 0.01, image resolution 128*128, and momentum = 0.9. We quickly realized that the number of epochs are too limited. We tried to increase the training epochs to see where the test accuracy stops to increase and where the loss stops to decrease. After some more experiments, we've decided that epochs = 20 is a good balance between good accuracy and training time. 

Later in the project, we experimented with other configurations to improve the test accuracy. The configurations include:

- Different models, such as the ResNet18, ResNet50, ResNet101, InceptionV4, and EfficientNetV1-B5
- Hyper-parameter settings such as the learning rate and the stepwise learning rate decay
- Using ImageNet pre-trained weights or none 
- Image resolutions: From 128 * 128 to 600 * 600
- K-fold cross validation: enabled or none
- Dropout in the FC layer: none, or p = 0.2, p=0.5
- Weight decay: none, or 0.0005
- Adaptive learning rate: decreasing the learning rate at scheduled epochs
- Stepwise learning rate decrease: slightly decrease the learning rate at every epoch

### What problems did you run into?

#### Bigger is better (to a limit)

The `resnet18` is a fairly small network by today's standards. Our testing accuracy plateaued at around 60%. To further improve the test accuracy, we then looked for more advanced neural networks and increased the resolution of input images.

We first tried bigger models in the ResNet family. We tried ResNet50 and ResNet101 at the same 128*128 resolution. The testing accuracy improved to around 70%. At this point, we couldn't specifically deduce whether the bottleneck was the small resolution or the relatively old ResNet network. Thus we ran tests with more modern networks and bigger image resolutions. 



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

### Why did you think this approach was better than other options?

- Using 10-fold cross validation, elaborate loss, accuracy, test accuracy difference

## Experiments

### Try multiple things to see what works better

- Different models on the same basic hyper-parameter setting: ResNet18, ResNet50, ResNet101, Inception V3, Inception V4, Efficient Net V1-B5.

- Diagrams

## Results

### What worked better

Cross Validation reduced loss.

### Maybe have some nice charts and graphs here

## Discussion

### What worked well and didn’t and WHY do you think that’s the case?

### Did you learn anything?

We found an empirical rule from the training:

- Small networks (such as InceptionV4) require less time for each epoch (when using the same GPU). However, they take more epochs to reduce their loss. 
- Bigger networks (such as EfficientNetV1-B5) require more time for each epoch. For instance, when using the same image resolution (512*512) and the maximum batch size as the Google Colab GPU permits, a typical EfficientNetV1-B5 epoch requires 60 minutes whereas an InceptionV4 epoch requires only 30 minutes. Nevertheless, Bigger networks requires less epochs to reduce the loss. 

### Can anything in this project apply more broadly to other projects?

