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
    
    Structure:
        The definition of function is the following:
            build_model_ INPUT SIZE _ ACTIVATION FUNCTION
            the mention cust_loss is used if the costum_loss is used instead 
        
        The one that has been selected in the report is the ..._cust_loss_sig
        In all architecture the neural network is optimize using Adam optimizer

"""


def build_model_40(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(40 ,input_shape=(40,)))        
    model.add(Activation('tanh'))
              
    model.add(Dense(40))        
    model.add(Activation('tanh'))
              
    model.add(Dense(40))      
    model.add(Activation('tanh'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

def build_model_22(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(22 ,input_shape=(22,)))        
    model.add(Activation('tanh'))
              
    model.add(Dense(22))        
    model.add(Activation('tanh'))
              
    model.add(Dense(22))      
    model.add(Activation('tanh'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

def build_model_10(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(10 ,input_shape=(10,)))        
    model.add(Activation('tanh'))
              
    model.add(Dense(10))        
    model.add(Activation('tanh'))
              
    model.add(Dense(10))      
    model.add(Activation('tanh'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model


def custum_loss(y_actual,y_pred)
    """
    new loss that allows to take the best orientatoin into account
    
    Args:
        y_actual: true target 
        y_pred: predicted target
    Results:
        loss value 
    """
    check = kb.sign(np.pi/2-y_pred)*np.pi
    val_1 = kb.square(y_actual-y_pred)
    val_2 = kb.square(y_actual-(y_pred+check))
    #val_3 = kb.square(y_actual-(y_pred-np.pi))
    #tmp = kb.minimum(val_3,val_2)
    tot = kb.minimum(val_1,val_2)
    return kb.mean(tot)

def build_model_40_cust_loss(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(40 ,input_shape=(40,)))        
    model.add(Activation('tanh'))
              
    model.add(Dense(40))        
    model.add(Activation('tanh'))
              
    model.add(Dense(40))      
    model.add(Activation('tanh'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss=custum_loss,optimizer='adam')
    return model

def build_model_40_cust_loss_relu(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(40 ,input_shape=(40,)))        
    model.add(Activation('relu'))
              
    model.add(Dense(40))        
    model.add(Activation('relu'))
              
    model.add(Dense(40))      
    model.add(Activation('relu'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss=custum_loss,optimizer='adam')
    return model

def build_model_40_cust_loss_sig(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(40 ,input_shape=(40,)))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(40))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(40))      
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss=custum_loss,optimizer='adam')
    return model

def build_model_24_cust_loss_sig(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(24 ,input_shape=(24,)))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(24))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(24))      
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss=custum_loss,optimizer='adam')
    return model
def build_model_60_cust_loss_sig(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(60 ,input_shape=(60,)))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(60))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(60))      
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss=custum_loss,optimizer='adam')
    return model
def build_model_84_cust_loss_sig(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(84 ,input_shape=(84,)))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(84))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(84))      
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss=custum_loss,optimizer='adam')
    return model

def build_model_40_cust_loss_batch(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(40 ,input_shape=(40,)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
              
    model.add(Dense(40))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
              
    model.add(Dense(40))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss=custum_loss,optimizer='adam')
    return model

def build_model_40_cust_loss_batch_sig(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(40 ,input_shape=(40,)))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
              
    model.add(Dense(40))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
              
    model.add(Dense(40))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss=custum_loss,optimizer='adam')
    return model



def build_model_40_cust_loss_deeper(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(40 ,input_shape=(40,)))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(40))        
    model.add(Activation('sigmoid'))
    
    model.add(Dense(40))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(40))      
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss=custum_loss,optimizer='adam')
    return model

def build_model_40_cust_loss_deeper_batch(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(40 ,input_shape=(40,)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
              
    model.add(Dense(40))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    
    model.add(Dense(40))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
              
    model.add(Dense(40))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss=custum_loss,optimizer='adam')
    return model



def build_model_40_cust_loss_deeper_2(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(80 ,input_shape=(40,)))        
    model.add(Activation('tanh'))
              
    model.add(Dense(80))        
    model.add(Activation('tanh'))
    
    model.add(Dense(80))        
    model.add(Activation('tanh'))
    
    model.add(Dense(80))        
    model.add(Activation('tanh'))
              
    model.add(Dense(80))      
    model.add(Activation('tanh'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss=custum_loss,optimizer='adam')
    return model

def build_model_40_cust_loss_deeper_2_batch(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(80 ,input_shape=(40,)))        
    model.add(Activation('tanh'))
              
    model.add(Dense(80))        
    model.add(Activation('tanh'))
    
    model.add(Dense(80))        
    model.add(Activation('tanh'))
    
    model.add(Dense(80))        
    model.add(Activation('tanh'))
              
    model.add(Dense(80))      
    model.add(Activation('tanh'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss=custum_loss,optimizer='adam')
    return model


def build_model_40_cust_loss_wider(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(80 ,input_shape=(40,))) 
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    
              
    model.add(Dense(80))  
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    
              
    model.add(Dense(80))  
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss=custum_loss,optimizer='adam')
    return model

def build_model_40_cust_loss_wider_batch(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(80 ,input_shape=(40,)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    
              
    model.add(Dense(80))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    
              
    model.add(Dense(80))  
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss=custum_loss,optimizer='adam')
    return model



def build_model_22_cust_loss(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(22 ,input_shape=(22,)))        
    model.add(Activation('tanh'))
              
    model.add(Dense(22))        
    model.add(Activation('tanh'))
              
    model.add(Dense(22))      
    model.add(Activation('tanh'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss=custum_loss,optimizer='adam')
    return model

def build_model_10_cust_loss(summary = False):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(10 ,input_shape=(10,)))        
    model.add(Activation('tanh'))
              
    model.add(Dense(10))        
    model.add(Activation('tanh'))
              
    model.add(Dense(10))      
    model.add(Activation('tanh'))
    model.add(Dense(1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss=custum_loss,optimizer='adam')
    return model


