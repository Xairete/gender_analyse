# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:33:44 2019

@author: gev
"""

import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
import python_speech_features as mfcc
from sklearn import preprocessing
import warnings

warnings.filterwarnings("ignore")
 
def get_MFCC(sr, audio):
    features = mfcc.mfcc(audio, sr, 0.025, 0.01, 10, appendEnergy = True)
    features = preprocessing.scale(features)
    return features

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
 
fname = ['female\\','male\\' ] #

for name in fname:
    #path to training data
    source   = "E:\\Other\\Projects\\pygender\\train_data\\youtube\\"+name
    #path to save trained model
    dest     = "E:\\Other\\Projects\\pygender\\"
    files    = [os.path.join(source,f) for f in os.listdir(source) if
                 f.endswith('.wav')]
    features = np.asarray(());
     
    for f in files:
        sr, audio = read(f)
        vector   = get_MFCC(sr,audio)
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))
     
    gmm = GMM(n_components = 8, max_iter = 200, covariance_type='diag', n_init = 3, verbose = 2,random_state = 10)
    gmm.fit(features)
    picklefile = f.split("\\")[-2].split(".wav")[0]+".gmm"
     
    # model saved as [name].gmm
    f = dest + picklefile
    cPickle.dump(gmm, open(f,'wb'))
    print ('modeling completed for gender:',picklefile)

modelpath  = "E:\\Other\\Projects\\pygender\\"    
gmm_files = [os.path.join(modelpath,fname) for fname in
              os.listdir(modelpath) if fname.endswith('.gmm')]
models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
genders   = [fname.split("\\")[-1].split(".gmm")[0] for fname
              in gmm_files]

#________________________________________________________________________________
sr_train, audio_train  = read("e:\\Other\\Projects\\pygender\\test_data\\test.wav")

t = moving_average(abs(audio_train), 20000)
t[t<0.0005] = 0
   
first = 0
end = 0
flag = 0
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
    print ("\tdetected as - ", genders[winner],"\n\tscores:female ",log_likelihood[0],",male ", log_likelihood[1],"\n")

