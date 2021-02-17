# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 10:00:00 2020

@author: Nadav
"""
# Constants variables
BATCH = 80
EPOCH = 10
ROWS = 224
COLS = 224
CHANNELS = 3


import tensorflow as tf
import os
import pickle
import random
import numpy as np
from result import model_results
import matplotlib.pyplot as plt  
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.image as mpimg
from PIL import Image
from keras.models import load_model
from keras.models import Sequential

            
def save_object(obj, file_name:str):
    """
    This method get dictionary that contain the name and instanse 
    of the model custom layers
    """
    with open(file_name, 'wb') as f:
            pickle.dump(obj.history, f)


def my_model(objective_function:str):
    # create the model - ResNet50
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(ROWS, COLS, CHANNELS),
        include_top=False,
        pooling = 'avg'
    )  # Do not include the ImageNet classifier at the top.
    x = tf.keras.layers.Dense(CLASSES, activation= 'softmax')
    
    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(x)
    
    # Freeze the base_model
    model.layers[0].trainable = False
    
    if objective_function == 'binary_crossentropy':
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(
                      from_logits=True, label_smoothing=0.1),
                  metrics=['accuracy'])
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
            
    return model
         

if __name__ == '__main__':  
    
    ## Open for tain dog_vs_cat model
    DIR_PATH = 'PetImages'
    model_name = 'dog_vs_cat_model.h5'
    history_object = 'dog_vs_cat_history.txt'
    class_mode = 'categorical'
    OBJECTIVE_FUNCTION = 'binary_crossentropy'
    CLASSES = 2
    
    ## Open for train flower model
    # DIR_PATH = 'flower_photos'
    # model_name = 'flower_model.h5'
    # history_object = 'flower_history.txt'
    # class_mode = 'sparse'
    # OBJECTIVE_FUNCTION = 'categorical_crossentropy'
    # CLASSES = 5
    
    data_dir = os.path.join(DIR_PATH, '')
    
    # Preprocessing pipelines
    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.4
        )
    
    # validation dataset
    val_ds = ImageDataGenerator.flow_from_directory(
        data_dir,
        target_size=(ROWS, COLS),
        seed=123,
        subset='validation',
        shuffle = True,
        class_mode = class_mode,
        )
    
    # class names
    class_names = val_ds.class_indices
    # size of val_ds
    size = len(val_ds.filepaths)
    
    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.4,
        horizontal_flip=True,
        brightness_range = [0.6, 1.4],
        )
    
    # training dataset
    train_ds = ImageDataGenerator.flow_from_directory(
        data_dir,
        target_size=(ROWS, COLS),
        seed=123,
        subset='training',
        shuffle = True,
        class_mode = class_mode,
        )
    
    # create the model
    model = my_model(OBJECTIVE_FUNCTION)
    # train the model
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCH,
                     validation_steps=10)
    
    # Save model
    tf.keras.models.save_model(model, model_name)
    
    # # save History object
    save_object(history, history_object)
