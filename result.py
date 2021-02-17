# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 10:00:00 2020

@author: Nadav
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def model_results(history, model, validation_ds, model_name:str):
    """
    This method create summarize plotting,
    and show the final evaluation of the model.
    """    
    if type(history) == dict:
        acc = history['accuracy']
        val_acc = history['val_accuracy']
        loss = history['loss']
        val_loss = history['val_loss']
    else:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
    
        
    fig, axes = plt.subplots(ncols=2, figsize=(10,5))
    
    plt.sca(axes[0])
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.gca().set_xlabel('Epoch', fontsize=14)
    plt.gca().set_ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1)
    plt.gca().set_title('Training and Validation Accuracy', 
                        fontstyle='italic', fontweight="bold") 
    plt.grid(True)

    
    plt.sca(axes[1])
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='lower right', fontsize=12)
    plt.gca().set_xlabel('Epoch', fontsize=14)
    plt.gca().set_ylabel('Loss', fontsize=14)
    if "flower" in model_name:
        plt.ylim(0.5, 1.6)
    else:
        plt.ylim(0, 1)
    plt.gca().set_title('Training and Validation Loss', 
                        fontstyle='italic', fontweight="bold")
    plt.grid(True)
    
    
    # save prediction_grid.png
    fig.savefig(model_name)
    # show to prediction_grid.png
    Image.open(model_name).show()    
    plt.show()
    
    # Final evaluation of the model   
    scores = model.evaluate(validation_ds, verbose=1)
    loss, accuracy = model.evaluate(validation_ds, verbose=1)
    if "flower" in model_name:
        write_to_file(scores, loss, accuracy, "flower_model_results.txt")
    else:
        write_to_file(scores, loss, accuracy, "dog_vs_cat_model_results.txt")


def write_to_file(scores, loss, accuracy, file_name:str):
    with open(file_name, 'w+') as f:
        f.write("CNN Error: %.2f%%\n" % (100-scores[1]*100))
        f.write("Loss: %.2f\n" %loss)
        f.write("Accuracy: %.0f%%\n" %(accuracy*100))


def make_prediction(val_ds, model, ROWS, COLS, prediction_grid_name:str):
    """
    This method create a grid of images, 
    with the true label and the prediction of each image.
    """
    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                     axes_pad=0.8  # pad between axes in inch.
                     )
    
    for i in range(9):
        test_img, label, img_for_plt = get_image_predict(val_ds, 
                        val_ds.class_indices, ROWS, COLS)
        pred = model.predict(test_img, verbose=1) 
        class_str = get_image_class(pred.argmax() ,val_ds.class_indices)
        grid[i].imshow(img_for_plt) 
        title = "True Label: {}\n Predicted: {}\n".format(label, class_str)
        grid[i].set_title(title)
        # grid[i].title(title)
        grid[i].axis("off")
    
    # show grid
    plt.show()
    # save prediction_grid.png
    fig.savefig(prediction_grid_name)
    # show to prediction_grid.png
    Image.open(prediction_grid_name).show()
    
    
def get_image_predict(val_ds, class_names, ROWS, COLS):
    """
    This methog get a validation dataset, and return 
    a numpy array that represent one of the images in the 
    validation dataset and it's label.
    Parameters
    ----------
    val_ds : is the validation dataset
    class_names : is a dictionary that contain all the class names

    Returns
    ----------
    Image from numpy Array of FP16

    """
    # get image for predict
    size = len(val_ds.filepaths)
    # Pick index in a random way
    index = random.randint(0,size-1)    
    image_path = val_ds.filepaths[index]
    
    img_for_plt = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=(180, 180)
        )
    
    select_img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=(ROWS, COLS)
        )
    select_img = tf.keras.preprocessing.image.img_to_array(select_img)
    select_img = np.array([select_img])  # Convert single image to a batch.
    
    # Get image label 
    val = val_ds.labels[index]
    label = get_image_class(val, class_names)
    
    return select_img, label, img_for_plt


def get_image_class(label_num, class_names):
    """
    This method return string class that represent the image.
    """
    for item in class_names.items():
        if item[1] == label_num:
            label_class = item[0]
            break
    return label_class


def create_roc_curve(model, test_ds):
    """
    This method create ROC curve to the dog_vs_cat model.
    """
    y_pred_keras = model.predict(test_ds)
    y_pred_keras = y_pred_keras[:,0]
    test_label = test_ds.labels
    
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_label, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)
    
    fig = plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate', fontsize=14)
    plt.ylabel('True positive rate', fontsize=14)
    plt.title('ROC curve - dog & cat model', fontstyle='italic', fontweight="bold")
    plt.legend(loc='best', fontsize=14)
    
    plt.show()
    # save prediction_grid.png
    fig.savefig("Dog_vs_Cat_Roc_Curve.png")
    # show to prediction_grid.png
    Image.open("Dog_vs_Cat_Roc_Curve.png").show()
        