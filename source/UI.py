'''
    Copyright (C),2018, Jiangshan Luo
    Filename: UI.py
    Author: Jiangshan Luo     Date: 12/10/2018
    Description:    User Interface of the program. Provide GUI for environment configurations.
    
    FunctionList:  
        1. load_config
        2. run_config
        3. get_params
        4. choose_video
'''

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from tkinter import *
from tkinter.filedialog import askopenfilename

class config:
    def __init__(self):
        self.config_form = Tk()
        self.threshold_config = None
        self.file_path = None
    
    def load_config(self):
        self.threshold_config = self.entry_1.get()
        print('threshold config:', self.threshold_config)
        self.config_form.destroy()

    def run_config(self):
        # window size
        width = 300
        height = 200
        # get screen width and height
        screenwidth = self.config_form.winfo_screenwidth() # width of the screen
        screenheight = self.config_form.winfo_screenheight() # height of the screen

        # calculate x and y coordinates for the Tk root window
        x = (screenwidth/2) - (width/2)
        y = (screenheight/2) - (height/2)

        # set the dimensions of the screen 
        # and where it is placed
        self.config_form.geometry('%dx%d+%d+%d' % (width, height, x, y))


        self.config_form.title('config window')
        Label(self.config_form, text="Ideal Threshold Value").grid(row=0)
        self.entry_1 = Entry(self.config_form)
        self.entry_1.grid(row=0, column=1)
        Button(self.config_form, text='Enter', command=self.load_config).grid(row=3, column=1, sticky=W, pady=4)
        
        mainloop()

    def get_params(self):
        return self.threshold_config

    def choose_video(self):
        self.file_path = askopenfilename()

        print('open file:', self.file_path)
        return self.file_path