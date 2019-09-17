# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:20:52 2019

@author: gev
"""
import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn import preprocessing
import tkinter 
from tkinter import filedialog as fd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def get_MFCC(sr,audio):
    features = mfcc.mfcc(audio, sr, 0.025, 0.01, 10,appendEnergy = True)
    features = preprocessing.scale(features)
    return features

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def LoadFile(): 
    fn = fd.askopenfilename(filetypes = [('*.wav files', '.wav')])
    print('fn', fn)
    if fn == '':
        return
    modelpath  = "E:\\Other\\Projects\\pygender\\"    
     
    gmm_files = [os.path.join(modelpath,fname) for fname in
              os.listdir(modelpath) if fname.endswith('.gmm')]
    models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
    genders   = [fname.split("\\")[-1].split(".gmm")[0] for fname
              in gmm_files]
    
    sr_train, audio_train  = read(fn)
    
    t = moving_average(abs(audio_train), 20000)
    t[t<0.0005] = 0
       
    first = 0
    end = 0
    flag = 0
    fig = Figure(figsize=(5, 4), dpi=100)
    fig.add_subplot(111).plot(audio_train, "-g")#
    while (flag == 0):
        new_first = np.where(t[end:]>0)
        if (new_first[0] != []):
            first = end + new_first[0][0]
        else:
            flag = 1
        new_end = np.where(t[first:]==0)
        if (new_end[0] != []):
            end = first + new_end[0][0]
        else:
            end = len(audio_train)
            flag = 1
            
        audio_slice = audio_train[first:end]
        
        features   = get_MFCC(sr_train, audio_slice)
        scores     = None
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm    = models[i]         #checking with each model one by one
            scores = np.array(gmm.score(features))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        if (winner == 0):
            args = '-g'
        else:
            args = '-y'
        print ("\tdetected as - ", genders[winner],"\n\tscores:female ",log_likelihood[0],",male ", log_likelihood[1],"\n")
        x = range(first,end)
        fig.add_subplot(111).plot(x, audio_train[first:end], args)#
    
    canv = FigureCanvasTkAgg(fig, master=root)
          
    canv.draw()
    canv.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

root = tkinter.Tk()

title = tkinter.Label(root, text="Gender parse")    #название программы
title.pack()    #упаковываем grid'ом, позицию элементов можно менять, изменяя параметры row и column

import_btn =tkinter.Button(root, text="Import file...", command=LoadFile)    #создаём кнопку
import_btn.pack(side=tkinter.BOTTOM)

fem_label = tkinter.Label(root, text="Female", fg="green")  
male_label = tkinter.Label(root, text="Male", fg="yellow")   
fem_label.pack(side=tkinter.LEFT)
male_label.pack(side=tkinter.RIGHT)


root.mainloop()