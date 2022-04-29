# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:44:00 2021

@author: Amir Modan

Main Program for Offline system which reads data, trains CNN model,
    and performs K-Fold Cross Validation

"""

from DataRetrieval import dataset
import tensorflow as tf
import time
import random
from CNN import trainData, testData

if __name__ == "__main__":
    #Fetch EMG Feature data for each gesture
    data1 = dataset("Dataset/HandGesture01.txt")
    label1 = [0 for i in range(len(data1))]
    data2 = dataset("Dataset/HandGesture02.txt")
    label2 = [1 for i in range(len(data2))]
    data3 = dataset("Dataset/HandGesture03.txt")
    label3 = [2 for i in range(len(data3))]
    data4 = dataset("Dataset/HandGesture04.txt")
    label4 = [3 for i in range(len(data4))]
    data5 = dataset("Dataset/HandGesture05.txt")
    label5 = [4 for i in range(len(data5))]
    data6 = dataset("Dataset/HandGesture06.txt")
    label6 = [5 for i in range(len(data6))]
    data7 = dataset("Dataset/HandGesture07.txt")
    label7 = [6 for i in range(len(data7))]
    data8 = dataset("Dataset/HandGesture08.txt")
    label8 = [7 for i in range(len(data8))]
    
    #Combine all data into one set, ignoring 4 gestures for better comparability
    data = data1 + data2 + data3 + data4
    label = label1 + label2 + label3 + label4
    
    #TODO: REMOVE RANDOMIZATION
    #Shuffle Dataset for K-Fold Validation
    temp = list(zip(data, label))
    random.shuffle(temp)
    data, label = zip(*temp)
    
    #Divide dataset so each fold has 100 samples
    divisor = 100
    x = [data[i:i + divisor] for i in range(0, len(data), divisor)]
    y = [label[i:i + divisor] for i in range(0, len(label), divisor)]
    init_time = time.time()
    total_acc = 0
    total_loss = 0
    for i in range(8):
        x_train = []
        y_train = []
        for j in range(i):
            x_train += x[j]
            y_train += y[j]
        for j in range(i+1,8):
            x_train += x[j]
            y_train += y[j]
        
        x_test = x[i]
        y_test = y[i]
        #Creates a new Sequential Neural Net.
        #"Sequential" refers to the fact that neural layers are stacked in sequence
        model = tf.keras.models.Sequential()
        model = trainData(model, x_train, y_train, 8, 4)
        loss, acc = testData(model, x_test, y_test, 8)
        total_acc += acc
        total_loss += loss
        
        
    print("Final loss: ", total_loss/8,"\nFinal accuracy: ", total_acc/8)
    print("Time taken to train Net: "+str(time.time()-init_time)+" seconds\n")