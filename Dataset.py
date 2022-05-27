# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:44:00 2021

@author: Amir Modan

Main Program for Offline system which reads data, trains CNN model,
    and performs K-Fold Cross Validation

"""

from DataRetrieval import dataset, extractFeatures
import tensorflow as tf
import time
from CNN import trainData, testData

# Instantiate Dataset
data = [None] * 8
label = [None] * 8

# Constants
num_channels = 8
num_gestures = 8
win_length = 10
win_increment = win_length

# For K-Fold Cross-Validation
num_folds = 8

if __name__ == "__main__":
    #Fetch EMG Feature data for each gesture
    for gesture in range(num_gestures):
        # Extract raw EMG data
        data[gesture] = dataset("Dataset/HandGesture0" + str(gesture+1) + ".txt", win_length, win_increment)
        # Extract MAV from raw data
        #data[gesture] = extractFeatures(data[gesture])
        label[gesture] = [gesture for i in range(len(data[gesture]))]
    
    # Start computation timing
    init_time = time.time()
    
    # Initialize evaluation metrics
    total_acc = 0
    total_loss = 0
    
    # Size of each fold for K-Fold cross validation
    divisor = int(len(data[1]) / num_folds)
    
    # Train and test a model for each of 8 folds
    for i in range(num_folds):
        
        # Instatitate empty lists for current fold
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        
        # Marks the start and end of fold used for testing
        test_start = i*divisor
        test_end = (i+1)*divisor
        
        for gesture in range(num_gestures):
            # All folds before current fold used for training
            x_train += data[gesture][0:test_start]
            y_train += label[gesture][0:test_start]
            
            # Current fold used for testing
            x_test += data[gesture][test_start:test_end]
            y_test += label[gesture][test_start:test_end]
            
            # All folds after current fold used for training
            x_train += data[gesture][test_end:]
            y_train += label[gesture][test_end:]
        
        #Creates a new Sequential Neural Net for this fold.
        #"Sequential" refers to the fact that neural layers are stacked in sequence
        model = tf.keras.models.Sequential()
        
        # Train model using raw sEMG image
        model = trainData(model, x_train, y_train, num_gestures, num_channels, window_length=win_length)
        # Evaluate current model and update overall evaluation with results
        loss, acc = testData(model, x_test, y_test)
        
        # Train model using features
        #model = trainData(model, x_train, y_train, 8, 8)
        # Evaluate current model and update overall evaluation with results
        #loss, acc = testData(model, x_test, y_test, 8)
        
        total_acc += acc
        total_loss += loss
        
    # Print overall evaluation results
    print("Final loss: ", total_loss/num_folds,"\nFinal accuracy: ", total_acc/num_folds)
    print("Time taken to train Net: "+str(time.time()-init_time)+" seconds\n")