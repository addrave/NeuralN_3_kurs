import os

from keras.preprocessing.image import ImageDataGenerator

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from sklearn import svm
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import GridSearchCV, train_test_split
import pickle


# np.set_printoptions(precision=2, suppress=True)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def n5():

    # model = keras.models.load_model('C:/Users/stoun/PycharmProjects/untitled3/model.h5')
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(160, 160, 3, )))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(160, 160),
        batch_size=64,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(160, 160),
        batch_size=64,
        class_mode='categorical')

    model.fit_generator(
        train_generator,
        steps_per_epoch=2677/64,
        epochs=200,
        verbose=2,
        validation_data=validation_generator,
        validation_steps=200/64)

    model.save('C:/Users/stoun/PycharmProjects/untitled3/model.h5')

n5()
