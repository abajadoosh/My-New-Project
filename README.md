# My-New-Project


# Project Title

Final project for the Building AI course

## Summary

Malignant melanoma is the deadliest form of skin cancer and has, among cancer types, one of the most rapidly increasing incidence rates in the world. Early diagnosis is crucial, since if detected early, its cure is simple. Diagnosis is typically performed manually by trained dermatologists who have the expertise to distinguish melanoma from benign skin lesions.


## Background

In light of the lethal consequences of melanoma, the necessity for an early detection method has grown Dermoscopy is currently utilized to diagnose melanoma skin lesions accurately. This is due to the fact that melanoma is more likely to recover if it is discovered early on. However, it is difficult to make a melanoma diagnosis using dermoscopic pictures due to the high level of expertise required. In order to automatically detect melanoma in dermoscopic images, deep learning models such as CNN and SVM are currently used. Different types of convolutional architectures and classification techniques are used in these existing CNN architectures, each with a different level of predictive accuracy. They all use CNN. The goal of this work is to develop a neural architecture that can accurately classify melanoma from photos captured in medical settings. Dermoscopic images are used in this study to develop a new CNN-based deep model for classifying skin lesions as either malignant or benign.

## How is it used?

Install Python
Install Libraries in Python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(11) # It's my lucky number
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import itertools

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
from tensorflow.keras.layers import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

Download Dataset from kaggle
Import Dataset in Python
Run the Code


This is how you create code examples:
```
def build(input_shape= (224,224,3), lr = 1e-3, num_classes= 2,
          init= 'normal', activ= 'relu', optim= 'adam'):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),padding = 'Same',input_shape=input_shape,
                     activation= activ, kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3),padding = 'Same', 
                     activation =activ, kernel_initializer = 'glorot_uniform'))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=init))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.summary()

    if optim == 'rmsprop':
        optimizer = RMSprop(lr=lr)

    else:
        optimizer = Adam(lr=lr)

    model.compile(optimizer = optimizer ,loss = "binary_crossentropy", metrics=["accuracy"])
    return model

main()
```


## DataSet sources
Dataset that we used in this thesis is consist of 2 kind of images, Malignant and Benign. Over Data set is divided into 2 categories, training data and testing. Furthermore, training dataset is divided into Malignant images and Benign images and same case is with testing dataset.

#!wget "https://storage.googleapis.com/kaggle-data-sets/174469/505351/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220412%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220412T193152Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=437dc15bf86519cc81d02c9d31316f8f23de86ff40f84d151cd1995d38b23f1ea5b066a29e35a4efed14961dd2736bc6e588066716be7e0095afb69a26514df3142c01055da88e560adc3b4c5c3a0f3c24940abfa0ef307ad0db57fb45920a8b0734f8a9c916ef7621e4a87eaceb7a5193747a4f96a9cfcc814e023cb25d1d212bf29262f06e0dfecdf1dbeaaf7267c0690e8c6e35769a76872940202e350b2a2fe84bafecbe656904d0b9bf2c3a8e4a6b03e44e000f14a0c387d47cdedbab071129b50bd75e29f2fa687c4c9c62c71eb47ed69d91c0a6e4a6166496e127e21480410da28a8bc3c63a969371bd71ad6a1f1f053d15aeb7d689ac94f9862b18a7"

## Aims

•	Improve the performance of deep learning on small datasets by developing new strategies.
•	Creation of an effective and advanced CNN classification method for the automatic classification of dermoscopy images of skin lesions
•	Dermoscopy images of skin lesions can be automatically classified using an efficient and powerful SVM algorithm.
