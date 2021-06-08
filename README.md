# Birdify

Aiming to be a better bird classification tool. 

Check out our project video: https://youtu.be/jaE03PImdGM

## Introduction

### What is the problem

We want to classify bird species based on bird images. We tested many different configurations that can affect the test accuracy to generate better predictions.

### Dataset

We use the [dataset](https://www.kaggle.com/c/birds21sp/data) provided on Kaggle.

The training set contains 555 species of birds, each with around 70 images. In total, the training set contains 38562 images.

The test set contains 10000 entries. Our current test accuracy is based on 10% of it.

### Team

Bowen Xu

Wenqing Lan

## Approach

### What techniques did we use?

We first started the training with the pretrained `resnet18` model. We also used the SGD optimizer and Cross Entropy Loss. (We compared the popular optimizers such as Adam, AdamW, and SGD. [This article](https://towardsdatascience.com/why-adamw-matters-736223f31b5d) shows that the SGD is still the optimizer that produces more generalizable models. Therefore we chose to use the SGD optimizer throughout the project.) We started our training with epochs = 5 and image resolution 128\*128. We quickly realized that the number of epochs are too limited. We tried to increase the training epochs to see where the test accuracy stops to increase and where the loss stops to decrease. After some more experiments, we've decided that epochs = 20 is a good balance between good accuracy and training time.

Later in the project, we experimented with other configurations to improve the test accuracy. The configurations include:

- Different models, such as the ResNet18, ResNet50, ResNet101, InceptionV4, and EfficientNetV1-B5
- Hyper-parameter settings such as the learning rate and the stepwise learning rate decay
- Using ImageNet pre-trained weights or none
- Input resolutions: From 128 x 128 to 600 x 600
- 10-fold cross validation: enabled or none
- Dropout in the FC layer: none, or p = 0.2, p=0.5
- Weight decay: none, or 0.0005
- Adaptive learning rate: decreasing the learning rate at scheduled epochs (`schedule={0: 0.09, 5: 0.01, 15: 0.001, 20: 0.0001, 30: 0.00001}`)
- Stepwise learning rate decreasing ratio: slightly decrease the learning rate with this ratio at every epoch

### What problems did we run into?

#### Finding the right network

The `resnet18` is a fairly small network by today's standards. Our testing accuracy plateaued at around 60%. To further improve the test accuracy, we then looked for more advanced neural networks and increased the resolution of input images.

We first tried bigger models in the ResNet family. We tried ResNet50 and ResNet101 at the same 128\*128 resolution. The testing accuracy improved to around 70%. At this point, we couldn't specifically deduce whether the bottleneck was the small resolution or the relatively old ResNet network. Thus we ran tests with more modern networks and bigger image resolutions.

The first bottleneck we met was with InceptionV4. While we were able to obtain around 80% accuracy, any further attempts to increase the resolution (and decrease the batch size) resulted in an intolerable training time: over 90 minutes per epoch. We then searched for more recent networks and found EfficientNetV1, one of the top-performing network on the [ImageNet benchmark](https://paperswithcode.com/sota/image-classification-on-imagenet). With 600\*600 input resolution, 10-fold cross validation, adaptive learning rate, and stepwise learning rate decrease, we were able to achieve 90.9% accuracy. You may find the full list of hyper-parameters [here](https://github.com/xubowenhaoren/Birdify/blob/9ef7acab532ae4e03923cd53a39b2800f17d1969/efficient_net_challenge.py#L16).

#### Overfitting

When we evaluated the training accuracy and loss logs from the above configuration, we noticed that the training accuracy reached 100% as early as epoch 1. This suggests overfitting and motivated us to compare the effectiveness of other techniques. See the experiments section below for more details.

#### Transfer learning: Time saver or bias maker?

So far we've trained our bird classifier with only pre-trained weights. However, we understand that the pre-trained weights come from ImageNet, which is a general classification problem. This is very different from our "specialized" bird classifier to differentiate different spices of birds. Did transfer learning introduce big bias that impeded the accuracy? We trained a new model without the ImageNet pre-trained weights. We found that the loss decreased very slowly and the resulting model performed poorly. (See the experiments section below for more details.) Therefore, we didn't use the brand-new EfficientNetV2 to train our bird classifier because there are no pre-trained weights available.

## Experiments

Note that due to the time limitations, we limited the input resolution to 512\*512. To avoid learning noisy data, we used the combination of exponential learning rate and adaptive learning rate. We also changed the stepwise learning rate decrease ratio per epoch from 97.5% to 90%.

#### K-fold cross validation

We evaluated the effectiveness of the 10-fold cross validation on both EfficientNetV1 and InceptionV4 through comparing with models of the same network trained without the cross validation.

|                                                                                                                                        |                                                                                                                               |
| :------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------: |
| <img src="graphs/EfficientNet with cross validation: Accuracy.png" alt="EfficientNet with cross validation: Accuracy" width="100%;" /> | <img src="graphs/EfficientNet with cross validation: Loss.png" alt="EfficientNet with cross validation: Loss" width="100%;"/> |

|                                                                                                                                                  |                                                                                                                                          |
| :----------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="graphs/EfficientNet with no cross validation: Accuracy.png" alt="EfficientNet with no cross validation: Accuracy" style="zoom:72%;" /> | <img src="graphs/EfficientNet with no cross validation: Loss.png" alt="EfficientNet with no cross validation: Loss" style="zoom:72%;" /> |

|                                                                                                                                          |                                                                                                                                  |
| :--------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------: |
| <img src="graphs/InceptionV4 with cross validation: Accuracy.png" alt="InceptionV4 with cross validation: Accuracy" style="zoom:72%;" /> | <img src="graphs/InceptionV4 with cross validation: Loss.png" alt="InceptionV4 with cross validation: Loss" style="zoom:72%;" /> |

|                                                                                                                                                |                                                                                                                                        |
| :--------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="graphs/InceptionV4 with no cross validation: Accuracy.png" alt="InceptionV4 with no cross validation: Accuracy" style="zoom:72%;" /> | <img src="graphs/InceptionV4 with no cross validation: Loss.png" alt="InceptionV4 with no cross validation: Loss" style="zoom:72%;" /> |

Both with and without cross validation, Efficient Net has higher training accuracy increasing rate and loss decreasing rate compared to Inception V4. We think it's because Efficient Net has more complicated internal structures. For each training epoch, Efficient Net takes 1 hour while Inception V4 takes 30 minutes.

#### No pre-trained weights

We evaluated the bias vs time-saving tradeoff of the transfer learning.

|                                                                                                                                                        |                                                                                                                                                |
| :----------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="graphs/EfficientNet with no pre-trained weights: Accuracy.png" alt="EfficientNet with no pre-trained weights: Accuracy" style="zoom:72%;" /> | <img src="graphs/EfficientNet with no pre-trained weights: Loss.png" alt="EfficientNet with no pre-trained weights: Loss" style="zoom:72%;" /> |

With no pre-trained weights, it's hard to achieve the same performance as the ones using pre-trained ones.

#### FC layer dropout

We evaluated the effect of having dropout in the fully connected (FC) layer.

TODO add plots for p = 0.2

|                                                                                                                                                                                |                                                                                                                                                                        |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="graphs/EfficientNet with cross-validation and dropout p=0.5: Accuracy.png" alt="EfficientNet with cross-validation and dropout p=0.5: Accuracy" style="zoom:72%;" /> | <img src="graphs/EfficientNet with cross-validation and dropout p=0.5: Loss.png" alt="EfficientNet with cross-validation and dropout p=0.5: Loss" style="zoom:72%;" /> |

|                                                                                                                                      |                                                                                                                              |
| :----------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------: |
| <img src="graphs/EfficientNet with dropout p=0.5: Accuracy.png" alt="EfficientNet with dropout p=0.5: Accuracy" style="zoom:72%;" /> | <img src="graphs/EfficientNet with dropout p=0.5: Loss.png" alt="EfficientNet with dropout p=0.5: Loss" style="zoom:72%;" /> |

Although using dropout rate achieved 100% training accuracy, it had high training loss and only had 0.1% test accuracy in the end.

#### Weight decay

We evaluated the effect of weight decay of 0.0005.

TODO add plots

## Results

Here are the test accuracies of the different configurations we tested. Note that we use N/T to abbreviate "not tested". Note that when the input resolution is 512\*512 unless otherwise specified.

| Configuration                            | With 10-fold cross validation | Without 10-fold cross validation |
| ---------------------------------------- | ----------------------------- | -------------------------------- |
| InceptionV4                              | 74.4%                         | 65.5%                            |
| EfficientNetV1                           | 86.9%                         | 89.2%                            |
| EfficientNetV1, no pre-trained weights   | 37.2%                         | N/T                              |
| EfficientNetV1, FC layer dropout p = 0.2 | TODO                          | TODO                             |
| EfficientNetV1, FC layer dropout p = 0.5 | 0.1%                          | 0.2%                             |
| EfficientNetV1, weight decay = 0.0005    | TODO                          | TODO                             |

## Discussion

### Evaluation

- What worked well and didn’t and Why do you think that’s the case? Why did we think this approach was better than other options?

Fixed parameters:

- epoch: 20
- momentum: 0.9

Worked well:

- Stepwise learning rate & exponential learning rate
  - It improved best training accuracy from 80% to 100%.
  - It improved minimum loss from 0.2 to as low as 0.0002.

Worked not well:

- Fully Connected Layer dropout p=0.2 and p=0.5
  - It lead to fluctuating training accuracy.
  - Loss only decreased to 2.
  - Test accuracy is as low as 0.1%.
- Weight decay = 0.0005
  - It lead to fluctuating training accuracy.
  - Loss remains high when comparing to other configurations.

Mix:

- Repeated 10-fold Cross Validation
  - As shown in the results, it improved test accuracy for InceptionV4 from 65.5% to 74.4%.
  - However, the test accuracy decreased from 89.2% to 86.9% after using repeated 10-fold Cross Validation.

### Conclusion

- The size of the network can have a big impact on the training time. Small networks (such as InceptionV4) require less time for each epoch (when using the same GPU). However, they take more epochs to reduce their loss. Bigger networks (such as EfficientNetV1-B5) require more time for each epoch. For instance, when using the same image resolution (512\*512) and the maximum batch size as the Google Colab GPU permits, a typical EfficientNetV1-B5 epoch requires 60 minutes whereas an InceptionV4 epoch requires only 30 minutes. Nevertheless, Bigger networks requires less epochs to reduce the loss.
- The law of diminishing returns apply to the tuning of hyper-parameters. During the training, we noticed that as we increase the number of epochs and the input resolution, the improvement of the test accuracy decreases.
- Having more appealing features doesn't always give you better results. Instead run experiments and pick the right ones for your network.
- Transfer learning can save you a lot of time. From our experiments, we know we can indeed reduce the number of epochs using pre-trained weights. At the same time, we can minimize the bias through fine-tuning with optimizers.

### Next steps

If we have more time on this project, we would try: 

- Changing the momentum
- Adding more controlled variables to each experiment set
- Extracting the output layer from our trained EfficientNetV1 model and use it as input for EfficientNetV2
  - Gain accuracy improvements through state-of-the-art networks

### Can anything in this project apply more broadly to other projects?

Yes! When new types of training data are available, our code can be used to generate other classifier models and are definitely not limited to birds. In addition, we've also added the following features that could be useful to gain insights about the training:

- Smart training progress bar. Instead of periodically calculating and predicting the loss, we introduced a smart progress bar that outputs the loss and the current iteration in real time. The current code also computes the training/validation accuracy once every 100 iterations and updates the progress bar.
- Automatic checkpointing, logging, and recovery. We used Google Colab to train our models. Since Google Colab has very strict usage limits, we've introduced automatic checkpoint recovery in case of GPU downtime. With this feature, our code automatically looks for the most recent checkpoint and automatically resumes the training. We've also added automatic logging to save the training loss and accuracy into an external file so that the relevant information are preserved even across notebook restarts. In our repo, we've also included code to parse the logs and generate the loss and accuracy graphs.

### References

We referred to Joe's [tutorial](https://www.kaggle.com/pjreddie/transfer-learning-to-birds-with-resnet18).
