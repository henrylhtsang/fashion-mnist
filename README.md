# fashion-mnist

This is an attempt in building a classification model with the fashion-mnist dataset. We build two models here, one with two dense layers of neural network, another with convolutional neural network. In both models, we use relu as the activation function, Adam as the optimizer, and SparseCategoricalCrossentropy as the loss function. 

In the deep neural network mode, we use two hidden layers, one of 100 nodes and the other of 30 nodes. At the end, we achieve an accuracy of 85%. The code is also available at [Kaggle](https://www.kaggle.com/henrylhtsang/fashion-mnist).

In the convolutional neural network, we use three convolutional layers and two dense layers. While the computation is much slower than the previous model, we achieve a 90% accuracy at the end. 
