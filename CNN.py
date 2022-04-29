# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:44:00 2021

@author: Amir Modan

Program containing methods for interacting with the Neural Network

"""

import tensorflow as tf 
import numpy as np

def trainData(model, train_data, train_label, num_features, num_classes):    
    """
    Trains an empty model

    Parameters
    ----------
    model : Sequential Object
        The model to be trained
    train_data : Array
        Data used to train model
    train_label : Array
        Labels used to supervise training of the model
    num_features : Integer
        The total number of features that will go into the network.
    num_classes: Integer
        The number of possible outcomes for classification

    Returns
    -------
    model
        The trained Neural Network model

    """    
    #Next Layers will utilize relu to activate the neuron
    model.add(tf.keras.layers.Conv1D(8, kernel_size=3, activation=tf.nn.relu,padding='same'))
    model.add(tf.keras.layers.Flatten())
    #Final Layer produces 4 possible outputs representing each gesture
    #Softmax Function assigns each output a probability so all outputs sum to 1
    model.add(tf.keras.layers.Dense(num_classes,activation=tf.nn.softmax))
    
    #Compile Neural Net with a learning rate of 0.01
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    
    train_label = np.array(train_label)
    train_data = np.array(train_data).reshape(-1, num_features, 1)
    #Train Neural Net through 2 epochs
    model.fit(train_data,train_label,epochs=2)
    
    return model

def testData(model, test_data, test_label, num_features):
    """
    Evaluates a trained model

    Parameters
    ----------
    model : Sequential Object
        The trained model to be tested
    test_data : Array
        Data used to test model
    test_label : Array
        Labels used to quantitate accuracy of model
    num_features : Integer
        The total number of features that will go into the network.
    

    Returns
    -------
    loss
        The calculated loss of the Neural Network
    accuracy
        The calculated accuracy of the Neural Network

    """
    test_label = np.array(test_label)
    test_data = np.array(test_data).reshape(-1, num_features, 1)
    loss, accuracy = model.evaluate(test_data,test_label)
    
    return loss, accuracy

def predict(model, test_data, num_features):
    """
    Makes a prediction based on a trained model

    Parameters
    ----------
    model : Sequential Object
        The trained model used for prediction
    test_data : Array
        Data to be used for prediction
    num_features : Integer
        The total number of features that will go into the network.
    

    Returns
    -------
    model.predict(test_data) : Array
        The prediction ratio for each class

    """
    test_data = np.array(test_data).reshape(-1, num_features, 1)
    return model.predict(test_data)


def preProcessing(mav1,mav2,mav3,mav4,mav5,mav6,mav7,mav8,zc1,zc2,zc3,zc4,zc5,zc6,zc7,zc8):
    """
    Takes features stored in separate arrays and appropriately combines each one to form a set of samples

    Parameters
    ----------
    mav(1-8) : Array
        Data pertaining to a single mav for a single channel.
    zc(1-8) : Array
        Data pertaining to a single zc for a single channel.

    Returns
    -------
    features : Array
        An array of samples where each sample contains mav and zc features for each channel

"""
    features = [[0.0 for i in range(16)] for j in range(len(mav1))]
    for i in range(len(mav1)):
        features[i] = [mav1[i], mav2[i], mav3[i], mav4[i], mav5[i], mav6[i], mav7[i], mav8[i], zc1[i], zc2[i], zc3[i], zc4[i], zc5[i], zc6[i], zc7[i], zc8[i]]
    return features