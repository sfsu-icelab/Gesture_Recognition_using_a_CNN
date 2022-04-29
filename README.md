# Gesture_Recognition_using_a_CNN
Initially designed as a course project for "ENGR 845 - Neural-Machine Interfaces", this is a Gesture Recognition Platform which classifies gestures using Surface Electromyography (sEMG) signals. The application is computer-based and was developed using Python. A Convolutional Neural Network (CNN) is used to classify these gestures in both offline and real-time applications.

Getting Started
---------------
To run either application, users must have downloaded the "Tensorflow" library which is used to define the Neural Network model. Instructions for downloading Tensorflow can be found here: https://www.tensorflow.org/install.<br/>
The "BLEAK" library is necessary for receiving sEMG data from a BLE device, namely the Myo Armband. Refer to https://bleak.readthedocs.io/en/latest/ for download instructions. This is only neccessary for executing the real-time application.<br/>
The "nest-asyncio" library is used to execute the Front-End GUI of the application asynchronously with the model and can be found here: https://pypi.org/project/nest-asyncio/. Once again, this is only needed for the real-time application.
