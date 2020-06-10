import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
import keras.backend as kb
from time import perf_counter


"""Model architecture

    Definition of all models architecture used in the code.
    The architecture corresponds to the best one selected in section discontinuity orientation
    
    Structure:
        The definition of function is the following:
            build_model_ INPUT SIZE
        
        The one that has been selected in the report is the ..._cust_loss_sig
        In all architecture the neural network is optimize using Adam optimizer

"""



def build_model_24(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(24 ,input_shape=(24,)))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(24))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(24))      
    model.add(Activation('sigmoid'))
    model.add(Dense(3))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

def build_model_40(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(40 ,input_shape=(40,)))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(40))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(40))      
    model.add(Activation('sigmoid'))
    model.add(Dense(3))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

def build_model_60(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(60 ,input_shape=(60,)))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(60))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(60))      
    model.add(Activation('sigmoid'))
    model.add(Dense(3))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

def build_model_84(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(84 ,input_shape=(84,)))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(84))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(84))      
    model.add(Activation('sigmoid'))
    model.add(Dense(3))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model