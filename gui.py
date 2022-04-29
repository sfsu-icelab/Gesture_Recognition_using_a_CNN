"""
@author: Darshanie Botejue

Program containing methods for defining Front-End layout of the Interface

"""

import tkinter as tk

#defining gui size
ht = 500 
wdth = 600

#defining button functions
def maintoinput():
	main_frame.place_forget()
	input_frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.3, anchor='n')

def maintooutput():
	main_frame.place_forget()
	output_frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.3, anchor='n')

def inputtomain():
	input_frame.place_forget()
	main_frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.3, anchor='n')

def outputtomain():
	output_frame.place_forget()
	main_frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.5, anchor='n')

#defining import function to take in input and convert to a string
def input_function():
    #print(entry.get())
    choice = entry.get()

#formatting gui
root = tk.Tk()
canvas = tk.Canvas(root, height = ht, width = wdth)
canvas.pack()

#creates main interface
main_frame = tk.Frame(root)
main_frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.1, anchor='n')

#creates input frame
input_frame =tk.Frame(root)

#creating input entry box and button to convert
entry = tk.Entry(input_frame)
entry.pack()

#creates output frame
output_frame = tk.Frame(root)
