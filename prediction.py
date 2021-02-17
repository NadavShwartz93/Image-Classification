# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 10:00:00 2020

@author: Nadav
"""
import tensorflow as tf
import pickle
import os
from result import model_results, make_prediction, create_roc_curve


# Constants variables
ROWS = 224
COLS = 224


def get_object(history_object:str):
    """
    This method return the History object that contain
    the next data: 'accuracy', 'val_accuracy', 'loss', 'val_loss' 
    of the model.
    """
    with open(history_object, "rb") as f:
        model_layers_dict = pickle.loads(f.read())
    return model_layers_dict


def create_test_dataset(dir_path, class_mode:str):
    """
    This method create test dataset, from images that located on disc,
    and return test dataset object.
    """
    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.4,
    )

    # test dataset
    test_ds = ImageDataGenerator.flow_from_directory(
        dir_path,
        target_size=(ROWS, COLS),
        seed=123,
        shuffle = True,
        subset='validation',
        class_mode = class_mode,
        )
    
    return test_ds

if __name__ == '__main__':  
    ## Open for make prediction for dog_vs_cat model
    # model_results_name = 'flower_model_results.png'
    # prediction_grid_name =  'flower_prediction_grid.png'
    # model_name = 'flower_model.h5'
    # history_object = 'flower_history.txt'
    # class_mode = 'sparse'
    # DIR_PATH = 'flower_photos'
    
    ## Open for make prediction for flower model
    model_results_name = 'Dog_vs_Cat_model_results.png'    
    prediction_grid_name = 'Dog_vs_Cat_prediction_grid.png'
    model_name = 'dog_vs_cat_model.h5'
    history_object = 'dog_vs_cat_history.txt'
    class_mode = 'categorical'
    DIR_PATH = 'PetImages'
       
    # load the trained model.
    model = tf.keras.models.load_model(model_name)
    
    # get History object.
    history = get_object(history_object)
    
    data_dir = os.path.join(DIR_PATH, '')
    test_ds = create_test_dataset(DIR_PATH, class_mode)
    
    # Show final results.
    model_results_name = os.path.join("Final_Images", model_results_name)
    model_results(history, model, test_ds, model_results_name)
    
    # # create grid of images with true labling and prediction.
    prediction_grid_name = os.path.join("Final_Images", prediction_grid_name)
    make_prediction(test_ds, model, ROWS, COLS, prediction_grid_name)
    
    if 'PetImages' in DIR_PATH:
        create_roc_curve(model, test_ds)
       