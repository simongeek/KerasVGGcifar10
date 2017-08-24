# Keras VGG implementation for cifar-10 classification

## What is Keras?

"Keras is an open source neural network library written in Python and capable of running on top of either [TensorFlow](https://www.tensorflow.org/), [CNTK](https://github.com/Microsoft/CNTK) or [Theano](http://deeplearning.net/software/theano/). 

Use Keras if you need a deep learning libraty that:
* Allows for easy and fast prototyping
* Supports both convolutional networks and recurrent networks, as well as combinations of the two
* Runs seamlessly on CPU and GPU

Keras is compatible with Python 2.7-3.5"[1].

Since Semptember 2016, Keras is the second-fastest growing Deep Learning framework after Google's Tensorflow, and the third largest after Tensorflow and Caffe[2].

## What is Deep Learning?

"Deep Learning is the application to learning tasks of artificial neural networks(ANNs) that contain more than one hidden layer. Deep learning is part of [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning) methods based on learning data representations.
Learning can be [supervised](https://en.wikipedia.org/wiki/Supervised_learning), parially supervised or [unsupervised](https://en.wikipedia.org/wiki/Unsupervised_learning)[3]."


## What will you learn?


You will learn:

* What is Keras library and how to use it
* What is Deep Learning
* How to use ready datasets
* What is Convolutional Neural Networks(CNN)
* How to build step by step Convolutional Neural Networks(CNN)
* What are differences in model results
* What is supervised and unsupervised learning
* Basics of Machine Learning
* Introduction to Artificial Intelligence(AI)

## Project structure

* vgg.py - simple example of VGG16 neural net
* README.md - description of this project

## Convolutional Neural Network
### VGG-16 neural network
#### Network Architecture

```
OPERATION           DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)

               Input   #####      3   32   32
       ZeroPadding2D   \|||/ -------------------         0     0.0%
                       #####      3   34   34
              Conv2D    \|/  -------------------      1792     0.0%
                relu   #####     64   32   32
       ZeroPadding2D   \|||/ -------------------         0     0.0%
                       #####     64   34   34
              Conv2D    \|/  -------------------     36928     0.1%
                relu   #####     64   32   32
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####     64   16   16
       ZeroPadding2D   \|||/ -------------------         0     0.0%
                       #####     64   18   18
              Conv2D    \|/  -------------------     73856     0.2%
                relu   #####    128   16   16
       ZeroPadding2D   \|||/ -------------------         0     0.0%
                       #####    128   18   18
              Conv2D    \|/  -------------------    147584     0.4%
                relu   #####    128   16   16
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####    128    8    8
       ZeroPadding2D   \|||/ -------------------         0     0.0%
                       #####    128   10   10
              Conv2D    \|/  -------------------    295168     0.9%
                relu   #####    256    8    8
       ZeroPadding2D   \|||/ -------------------         0     0.0%
                       #####    256   10   10
              Conv2D    \|/  -------------------    590080     1.8%
                relu   #####    256    8    8
       ZeroPadding2D   \|||/ -------------------         0     0.0%
                       #####    256   10   10
              Conv2D    \|/  -------------------    590080     1.8%
                relu   #####    256    8    8
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####    256    4    4
       ZeroPadding2D   \|||/ -------------------         0     0.0%
                       #####    256    6    6
              Conv2D    \|/  -------------------   1180160     3.5%
                relu   #####    512    4    4
       ZeroPadding2D   \|||/ -------------------         0     0.0%
                       #####    512    6    6
              Conv2D    \|/  -------------------   2359808     7.0%
                relu   #####    512    4    4
       ZeroPadding2D   \|||/ -------------------         0     0.0%
                       #####    512    6    6
              Conv2D    \|/  -------------------   2359808     7.0%
                relu   #####    512    4    4
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####    512    2    2
       ZeroPadding2D   \|||/ -------------------         0     0.0%
                       #####    512    4    4
              Conv2D    \|/  -------------------   2359808     7.0%
                relu   #####    512    2    2
       ZeroPadding2D   \|||/ -------------------         0     0.0%
                       #####    512    4    4
              Conv2D    \|/  -------------------   2359808     7.0%
                relu   #####    512    2    2
       ZeroPadding2D   \|||/ -------------------         0     0.0%
                       #####    512    4    4
              Conv2D    \|/  -------------------   2359808     7.0%
                relu   #####    512    2    2
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####    512    1    1
             Flatten   ||||| -------------------         0     0.0%
                       #####         512
               Dense   XXXXX -------------------   2101248     6.2%
                relu   #####        4096
             Dropout    | || -------------------         0     0.0%
                       #####        4096
               Dense   XXXXX -------------------  16781312    49.9%
                relu   #####        4096
             Dropout    | || -------------------         0     0.0%
                       #####        4096
               Dense   XXXXX -------------------     40970     0.1%
             softmax   #####          10
```

#### Model

```
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=x_train.shape[1:]))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.1, decay=1e-6, nesterov=True)

```
#### Train model:
```
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


cnn_n = base_model()
cnn_n.summary()
```

#### Fit model:
```
cnn = cnn_n.fit(x_train,y_train, batch_size=batch_size, epochs=epochs,validation_data=(x_test,y_test),shuffle=True)
```

#### Results:

All results are for 50k iteration, learning rate=0.01. Neural networks have benn trained at 16 cores and 16GB RAM on [plon.io](https://plon.io/).

* epochs = 10 **accuracy=10.0%**

[x](https://plon.io/files/599089b3fa48b400014a1974)
[y](https://plon.io/files/599089b4fa48b400014a1976)

**Confusion matrix result:**

```
[[   0    0    0    0    0    0    0    0    0 1000]
 [   0    0    0    0    0    0    0    0    0 1000]
 [   0    0    0    0    0    0    0    0    0 1000]
 [   0    0    0    0    0    0    0    0    0 1000]
 [   0    0    0    0    0    0    0    0    0 1000]
 [   0    0    0    0    0    0    0    0    0 1000]
 [   0    0    0    0    0    0    0    0    0 1000]
 [   0    0    0    0    0    0    0    0    0 1000]
 [   0    0    0    0    0    0    0    0    0 1000]
 [   0    0    0    0    0    0    0    0    0 1000]]
```

**Confusion matrix vizualizing**

Time of learning process: **xxx**

* epochs = 20 **accuracy=%**

**Confusion matrix result:**

**Confusion matrix vizualizing**

Time of learning process: **xxx**


* epochs = 50 **accuracy=%**

**Confusion matrix result:**

**Confusion matrix vizualizing**

Time of learning process: **xxx**

* epochs = 100 **accuracy=%**

**Confusion matrix result:**

**Confusion matrix vizualizing**

Time of learning process: **xxx**


.
## Resources

## Grab the code or run project in online IDE


You can also check [here](https://github.com/simongeek/KerasT) my other Keras Cifar-10 classification using 4,6-layer neural nets.
