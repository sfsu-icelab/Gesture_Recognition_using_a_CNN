# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:22:21 2021

@author: Amir Modan

Program containing methods for retrieving data from dataset

"""

def dataset(fileAddress):
    """
    Scans a .txt file for sEMG data and extracts features for each channel

    Parameters
    ----------
    fileAddress: String
        The location of the file to be read on the file explorer
        Must end with '.txt'

    Returns
    -------
    features: List (List (float))
    """
    #Opens a .txt file
    file = open(fileAddress,"r")
    #Splits lines into a String Array
    content = file.read().splitlines()
    #Closes the file
    file.close()
    data = [[0.0 for i in range(8)] for j in range(5000)]
    
    current_window = -100
    #Loops through each line and assigns to data array
    for str in content:
        #If line is empty, indicates current window is complete
        if(len(str) == 0):
            current_window = current_window + 100;
            continue
        #Indicates a new window of data
        elif(str[0:2] == '{{'):
            channel = 0
            string_arr = str.replace('{', '').replace('}', '').replace(',', '').split()
            for i in range(len(string_arr)):
                data[current_window+i][channel] = int(string_arr[i])
            channel = channel + 1
        #Indicates the next channel
        elif(str[0] == '{'):
            string_arr = str.replace('{', '').replace('}', '').replace(',', '').split()
            for i in range(len(string_arr)):
                data[current_window+i][channel] = int(string_arr[i])
            channel = channel + 1
            
    #Extract MAV for 8 sample window
    windows = [data[i:i + 8] for i in range(0, len(data), 8)]
    features = [[0.0 for i in range(8)] for j in range(len(windows))]
    for i in range(len(windows)):
        numsum = [0] * len(windows[i])
        for sample in windows[i]:
            for channel in range(len(sample)):
                numsum[channel] += abs(sample[channel])
        for channel in range(8):
            features[i][channel] = numsum[channel]/len(windows[i])
    return features