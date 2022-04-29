"""
@author: Amir Modan

Main Program for Real-Time system which establishes BLE connection,
    defines GUI, and integrates with CNN.

"""

import asyncio
import nest_asyncio
nest_asyncio.apply()
from typing import Any
from CNN import trainData, predict, preProcessing
from gui import *
from bleak import BleakClient, discover
import tensorflow as tf 
import warnings
warnings.filterwarnings("ignore")

#UUID's for BLE Connection
CONTROL = "d5060401-a904-deb9-4748-2c7f4a124842"
EMG0 = "d5060105-a904-deb9-4748-2c7f4a124842"
EMG1 = "d5060205-a904-deb9-4748-2c7f4a124842"
EMG2 = "d5060305-a904-deb9-4748-2c7f4a124842"
EMG3 = "d5060405-a904-deb9-4748-2c7f4a124842"

#Samples to be recored for each gesture
SAMPLES_PER_GESTURE = 20

#List of Gestures to be used for classification
GESTURES = ["Relaxation", "Fist", "Wave In", "Wave Out"]

mav1 = []
mav2 = []
mav3 = []
mav4 = []
mav5 = []
mav6 = []
mav7 = []
mav8 = []
zc1 = []
zc2 = []
zc3 = []
zc4 = []
zc5 = []
zc6 = []
zc7 = []
zc8 = []
label = []
current_sample = [0.0 for i in range(16)]
selected_device = []

#Generate initial label for GUI
label1 = tk.Label(output_frame, text = "Train Net First")
label1.pack()

class Connection:
    
    client: BleakClient = None
    
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        EMG0: str,
        EMG1: str,
        EMG2: str,
        EMG3: str,
        CONTROL: str,
    ):
        self.loop = loop
        self.EMG0 = EMG0
        self.EMG1 = EMG1
        self.EMG2 = EMG2
        self.EMG3 = EMG3
        self.CONTROL = CONTROL
        self.connected = False
        self.connected_device = None
        self.model = tf.keras.models.Sequential()

    """
        Handler for when BLE device is disconnected
    
    """
    def on_disconnect(self, client: BleakClient):
        self.connected = False
        print(f"Disconnected from {self.connected_device.name}!")

    """
        Callback right after BLE device is deisconnected
    
    """
    async def cleanup(self):
        #Terminates all communication attempts with BLE device
        if self.client:
            await self.client.stop_notify(EMG0)
            await self.client.stop_notify(EMG1)
            await self.client.stop_notify(EMG2)
            await self.client.stop_notify(EMG3)
            await self.client.disconnect()

    """
        Searches for nearby BLE devices or initiates connection with BLE device if chosen
    
    """
    async def manager(self):
        print("Starting connection manager.")
        while True:
            if self.client:
                await self.connect()
            else:
                await self.select_device()
                await asyncio.sleep(2.0, loop=loop)       
    
    """
        Performs initial actions on connection with BLE device, including training neural network
    
    """
    async def connect(self):
        if self.connected:
            return
        try:
            await self.client.connect()
            self.connected = await self.client.is_connected()
            if self.connected:
                print(F"Connected to {self.connected_device.name}")
                self.client.set_disconnected_callback(self.on_disconnect)
                
                #Must send below command to Myo Armband to initiate EMG communication
                bytes_to_send = bytearray([1, 3, 2, 0, 0])
                await connection.client.write_gatt_char(CONTROL, bytes_to_send)
                
                #Generates GUI label for gesture instruction
                label5 = tk.Label(output_frame, text = "Perform ")
                
                #Loops through each gesture and collects training data
                for i in range(len(GESTURES)):
                    print("Perform " + GESTURES[i])
                    #Updates GUI label for current gesture
                    label5.config(text = "Perform " + GESTURES[i])
                    label5.pack()
                    initial_length = len(mav8)
                    #Generate slight delay to allow time for user to perform next gesture
                    await asyncio.sleep(2.0, loop=loop)
                    
                    #Begin collecting training data
                    await self.client.start_notify(self.EMG0, self.training_handler0)
                    await self.client.start_notify(self.EMG1, self.training_handler1)
                    await self.client.start_notify(self.EMG2, self.training_handler2)
                    await self.client.start_notify(self.EMG3, self.training_handler3)
                    
                    #Continue until enough data is collected
                    while((len(mav8)-initial_length) < SAMPLES_PER_GESTURE):
                        await asyncio.sleep(0.05, loop=loop)
                        
                    #Stop collecting training data
                    await self.client.stop_notify(EMG0)
                    await self.client.stop_notify(EMG1)
                    await self.client.stop_notify(EMG2)
                    await self.client.stop_notify(EMG3)
                    
                    #If some channels sent more data than others,
                    #discards extra data so all channels have the same amount of data
                    minLength = min([len(mav1),len(mav3),len(mav5),len(mav7)])
                    if(len(mav1) > minLength):
                        del(mav1[-1])
                        del(zc1[-1])
                        del(mav2[-1])
                        del(zc2[-1])
                    if(len(mav3) > minLength):
                        del(mav3[-1])
                        del(zc3[-1])
                        del(mav4[-1])
                        del(zc4[-1])
                    if(len(mav5) > minLength):
                        del(mav5[-1])
                        del(zc5[-1])
                        del(mav6[-1])
                        del(zc6[-1])
                    if(len(mav7) > minLength):
                        del(mav7[-1])
                        del(zc7[-1])
                        del(mav8[-1])
                        del(zc8[-1])
                    
                    #Adds appropriate number of labels needed for supervision
                    label.extend([i for x in range(len(mav8)-initial_length)])
                
                #Reformats all data to be in samples
                features = preProcessing(mav1,mav2,mav3,mav4,mav5,mav6,mav7,mav8,zc1,zc2,zc3,zc4,zc5,zc6,zc7,zc8)

                #Trains Neural Network
                self.model = trainData(self.model, features, label, 16, len(GESTURES))
                
                #Updates GUI label to signal completion
                label5.config(text = "Training Complete")
                
                #Predict gestures until network is disconnected
                while True:
                    if not self.connected:
                        break
                    await self.client.start_notify(self.EMG0, self.prediction_handler0)
                    await self.client.start_notify(self.EMG1, self.prediction_handler1)
                    await self.client.start_notify(self.EMG2, self.prediction_handler2)
                    await self.client.start_notify(self.EMG3, self.prediction_handler3)
                    await asyncio.sleep(3.0, loop=loop)
            else:
                print(f"Failed to connect to {self.connected_device.name}")
        except Exception as e:
            print(e)

    """
        Selects and connects to a BLE device
    
    """
    async def select_device(self):
        print("Bluetooh LE hardware warming up...")
        await asyncio.sleep(2.0, loop=loop)
        #Searches for BLE devices
        devices = await discover()
        #Display all available BLE devices
        ble_devices = "\n"
        print("Please select device: ")
        for i, device in enumerate(devices):
            print(f"{i}: {device.name}")
            ble_devices += f"{i}: {device.name}\n"
        label2 = tk.Label(input_frame, text = ble_devices)
        label2.pack()
        
        #Prompt user to select BLE device until a valid choice has been made
        while True:
            response = -1
            while response == -1:
                root.update()
                if(len(entry.get()) > 0):
                    response = entry.get()
            #For CLI only
            #response = input("Select device: ")
            
            #If choice is invalid, prompt again.
            try:
                response = int(response.strip())
            except:
                print("Please make valid selection.")
            
            if response > -1 and response < len(devices):
                break
            else:
                print("Please make valid selection.")

        print(f"Connecting to {devices[response].name}")
        self.connected_device = devices[response]
        self.client = BleakClient(devices[response].address, loop=self.loop)

    #Handler for collecting data from Channels 1 and 2
    def training_handler0(self, sender: str, data: Any):
        num1a, num1b, num2a, num2b = getFeatures(data)
        mav1.append(num1a)
        zc1.append(num1b)
        mav2.append(num2a)
        zc2.append(num2b)
        
    #Handler for collecting data from Channels 3 and 4
    def training_handler1(self, sender: str, data: Any):
        num1a, num1b, num2a, num2b = getFeatures(data)
        mav3.append(num1a)
        zc3.append(num1b)
        mav4.append(num2a)
        zc4.append(num2b)
        
    #Handler for collecting data from Channels 5 and 6
    def training_handler2(self, sender: str, data: Any):
        num1a, num1b, num2a, num2b = getFeatures(data)
        mav5.append(num1a)
        zc5.append(num1b)
        mav6.append(num2a)
        zc6.append(num2b)
        
    #Handler for collecting data from Channels 7 and 8
    def training_handler3(self, sender: str, data: Any):
        num1a, num1b, num2a, num2b = getFeatures(data)
        mav7.append(num1a)
        zc7.append(num1b)
        mav8.append(num2a)
        zc8.append(num2b)
        
    #Handler for collecting data from Channels 1 and 2 for prediction
    def prediction_handler0(self, sender: str, data: Any):
        current_sample[0], current_sample[8], current_sample[1], current_sample[9] = getFeatures(data)
    #Handler for collecting data from Channels 3 and 4 for prediction
    def prediction_handler1(self, sender: str, data: Any):
        current_sample[2], current_sample[10], current_sample[3], current_sample[11] = getFeatures(data)
    #Handler for collecting data from Channels 5 and 6 for prediction
    def prediction_handler2(self, sender: str, data: Any):
        current_sample[4], current_sample[12], current_sample[5], current_sample[13] = getFeatures(data)
    #Handler for collecting data from Channels 7 and 8 for prediction and makes a prediction off the complete sample
    def prediction_handler3(self, sender: str, data: Any):
        current_sample[6], current_sample[14], current_sample[7], current_sample[15] = getFeatures(data)
        print(GESTURES[getMax(predict(self.model, current_sample, len(current_sample))[0])])
        label1.config(text = GESTURES[getMax(predict(self.model, current_sample, len(current_sample))[0])])
        root.update()
        #print(current_sample)
    

#############
# Loops
#############



async def main():
    #creating main frame to interface between wanting input and output
    button0= tk.Button(main_frame, text = "Click to select BLE Device", command = maintoinput)
    button1= tk.Button(main_frame, text = "Click to predict Gesture", command = maintooutput)
    button0.pack()
    button1.pack()
    
    button2 = tk.Button(input_frame, text = "Select Device", command = input_function)
    button3 = tk.Button(input_frame, text = "Return to main", command = inputtomain)
    button2.pack()
    button3.pack()        
    
    #creating output box and button
    button5= tk.Button(output_frame, text = "Return to main", command = outputtomain)
    button5.pack()
    
    #Continuously updates GUI
    while True:
        root.update()
        await asyncio.sleep(0.05, loop=loop)

"""
    Extracts features from 2 channels of data
    
    Parameters
    ----------
    data : Array
        Several samples of data taken for two channels

    Returns
    -------
    nummav1
        mav taken for a window of data from Channel 1
    numzc1
        zc taken for a window of data from Channel 1
    nummav2
        mav taken for a window of data from Channel 2
    numzc2
        zc taken for a window of data from Channel 2

"""
def getFeatures(data):
    numsum1 = 0
    numsum2 = 0
    numzc1 = 0
    numzc2 = 0
    for i in range(8):
        if data[i] > 127:
            numsum1 += (256-data[i])
        else:
            numsum1 += data[i]
        if i < 7 and ((data[i] < 127 and data[i+1] > 127) or (data[i] > 127 and data[i+1] < 127)):
            numzc1 += 1
            
    for i in range(8,16):
        if data[i] > 127:
            numsum2 += (256-data[i])
        else:
            numsum2 += data[i]
        if i < 15 and ((data[i] < 127 and data[i+1] > 127) or (data[i] > 127 and data[i+1] < 127)):
            numzc2 += 1
            
    
    nummav1 = numsum1 / (len(data)*63.5)
    nummav2 = numsum2 / (len(data)*63.5)
    return nummav1, numzc1/7, nummav2, numzc2/7

"""
    Gets index of maximum value, used for getting predicted gesture

    Parameters
    ----------
    arr : Array
        Probability assigned to each gesture
        
    maxIndex : Integer
        Index containing maximum probability, our prediction
    
"""
def getMax(arr):
    maxIndex = 0
    for i in range(len(arr)):
        if arr[i] > arr[maxIndex]:
            maxIndex = i
    return maxIndex

#############
# App Main
#############
if __name__ == "__main__":

    # Create the event loop.
    loop = asyncio.get_event_loop()
    connection = Connection(loop, EMG0, EMG1, EMG2, EMG3, CONTROL)
    try:
        asyncio.ensure_future(connection.manager())
        asyncio.ensure_future(main())
        loop.run_forever()
    except KeyboardInterrupt:
        print()
        print("User stopped program.")
    finally:
        print("Disconnecting...")
        loop.run_until_complete(connection.cleanup())
