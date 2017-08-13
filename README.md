# Keras VGG implementation for cifar-10 clasification


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

####Model

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
Train model:
```
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


cnn_n = base_model()
cnn_n.summary()
```

Fit model:
```
cnn = cnn_n.fit(x_train,y_train, batch_size=batch_size, epochs=epochs,validation_data=(x_test,y_test),shuffle=True)
```

Results:

All results are for 50k iteration, learning rate=0.01. Neural networks have benn trained at 16 cores and 16GB RAM on [plon.io](https://plon.io/).

* epochs = 10 **accuracy=%**

**Confusion matrix result:**

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
