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

### What techniques did we use?

We first started the training with the pretrained `resnet18` model. We also used the SGD optimizer and Cross Entropy Loss. (We compared the popular optimizers such as Adam, AdamW, and SGD. [This article](https://towardsdatascience.com/why-adamw-matters-736223f31b5d) shows that the SGD is still the optimizer that produces more generalizable models. Therefore we chose to use the SGD optimizer throughout the project.) We started our training with epochs = 5 and image resolution 128*128. We quickly realized that the number of epochs are too limited. We tried to increase the training epochs to see where the test accuracy stops to increase and where the loss stops to decrease. After some more experiments, we've decided that epochs = 20 is a good balance between good accuracy and training time. 

Later in the project, we experimented with other configurations to improve the test accuracy. The configurations include:

- Different models, such as the ResNet18, ResNet50, ResNet101, InceptionV4, and EfficientNetV1-B5
- Hyper-parameter settings such as the learning rate and the stepwise learning rate decay
- Using ImageNet pre-trained weights or none 
- Input resolutions: From 128 * 128 to 600 * 600
- K-fold cross validation: enabled or none
- Dropout in the FC layer: none, or p = 0.2, p=0.5
- Weight decay: none, or 0.0005
- Adaptive learning rate: decreasing the learning rate at scheduled epochs
- Stepwise learning rate decrease: slightly decrease the learning rate at every epoch

### What problems did we run into?

#### Finding the right network

The `resnet18` is a fairly small network by today's standards. Our testing accuracy plateaued at around 60%. To further improve the test accuracy, we then looked for more advanced neural networks and increased the resolution of input images.

We first tried bigger models in the ResNet family. We tried ResNet50 and ResNet101 at the same 128*128 resolution. The testing accuracy improved to around 70%. At this point, we couldn't specifically deduce whether the bottleneck was the small resolution or the relatively old ResNet network. Thus we ran tests with more modern networks and bigger image resolutions. 

The first bottleneck we met was with InceptionV4. While we were able to obtain 89% accuracy, any further attempts to increase the resolution (and decrease the batch size) resulted in an intolerable training time: over 90 minutes per epoch. We then searched for more recent networks and found EfficientNetV1, one of the top-performing network on the [ImageNet benchmark](https://paperswithcode.com/sota/image-classification-on-imagenet). With 600*600 input resolution, 10-fold cross validation, adaptive learning rate, and stepwise learning rate decrease, we were able to achieve 90.9% accuracy. You may find the full list of hyper-parameters [here](https://github.com/xubowenhaoren/Birdify/blob/9ef7acab532ae4e03923cd53a39b2800f17d1969/efficient_net_challenge.py#L16). 

#### Overfitting

When we evaluate the training accuracy and loss logs of the above configuration, we noticed that the training accuracy reached 100% as early as epoch 1. This suggests overfitting and motivated us to compare the effectiveness of other techniques. Note that due to the time limitations, we limited the input resolution to 512*512. 

- We changed the adaptive learning rate schedule to start with 0.09 instead of 0.01. 



#### Transfer learning: Time saver or bias maker?

So far we've trained our bird classifier with only pre-trained weights. However, we understand that the pre-trained weights come from ImageNet, which is a general classification problem. This is very different from our "specialized" bird classifier to differentiate different spices of birds. Did transfer learning introduce big bias that impeded the accuracy? We trained a new model without the ImageNet pre-trained weights. We found that the loss decreased very slowly and the resulting model performed poorly. (See the detailed plots below.) We can indeed same time using pre-trained weights, and minimize the bias through the use of fine-tuning with optimizers. 

### Why did we think this approach was better than other options?

## Experiments

- Different models on the same basic hyper-parameter setting: ResNet18, ResNet50, ResNet101, Inception V3, Inception V4, Efficient Net V1-B5.

- Diagrams

## Results

### What worked better

Cross Validation reduced loss.

### Maybe have some nice charts and graphs here

## Discussion

### Evaluation

- What worked well and didn’t and Why do you think that’s the case?

### Conclusion

- The size of the network can have a big impact on the training time. Small networks (such as InceptionV4) require less time for each epoch (when using the same GPU). However, they take more epochs to reduce their loss. Bigger networks (such as EfficientNetV1-B5) require more time for each epoch. For instance, when using the same image resolution (512*512) and the maximum batch size as the Google Colab GPU permits, a typical EfficientNetV1-B5 epoch requires 60 minutes whereas an InceptionV4 epoch requires only 30 minutes. Nevertheless, Bigger networks requires less epochs to reduce the loss. 

- The law of diminishing returns apply to the tuning of hyper-parameters. During the training, we noticed that as we increase the number of epochs and the input resolution, the improvement of the test accuracy decreases. 
- Having more appealing features doesn't always give you better results. Instead run experiments and pick the right ones for your network. 
- Transfer learning can save you a lot of time.

### Can anything in this project apply more broadly to other projects?

Yes! When new types of training data are available, our code can be used to generate other classifier models and are definitely not limited to birds. In addition, we've also added the following features that could be useful to gain insights about the training:

- Smart training progress bar. Instead of periodically calculating and predicting the loss, we introduced a smart progress bar that outputs the loss and the current iteration in real time. The current code also computes the training/validation accuracy once every 100 iterations and updates the progress bar. 
- Automatic checkpointing, logging, and recovery. We used Google Colab to train our models. Since Google Colab has very strict usage limits, we've introduced automatic checkpoint recovery in case of GPU downtime. With this feature, our code automatically looks for the most recent checkpoint and automatically resumes the training. We've also added automatic logging to save the training loss and accuracy into an external file so that the relevant information are preserved even across notebook restarts. In our repo, we've also included code to parse the logs and generate the loss and accuracy graphs. 

