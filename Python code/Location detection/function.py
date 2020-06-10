
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
from time import perf_counter 

""" Presentation of the functions used in the code 

Section:
    A. Data manipulation 
    B. Training and testing of the neural network

Note:
    The implementatoin of the neural networks use the Keras package:
        https://keras.io/api/
    Also it use tensorflow as backend
    The code also use the Numpy package :
        https://numpy.org/doc/1.18/reference/index.html

"""

##################### A. Data manipulation #############################
def data_cut( data , nb_samples):
     """ 
    Take a subsample of the data
    
    Args:
        data: Samples of data 
        nb_samples (int): size of the subsample
        
    Returns:
        The subsample of data
        
    """
    num_observation = np.shape(data)[0]
    random_permuted_indices = np.random.permutation(num_observation)
    data_suffle = data[random_permuted_indices]
    return data_suffle[ : nb_samples]

def data_split(data, ratio_train, cut ):
    """ 
    Split the data into train and test set
    
    Args:
        data: Samples of data 
        ratio_train (double): ratio of training sample
        cut (int): where to cut
        
    Returns:
        X_train: subset of input for training 
        X_test: subset of input for testing 
        Y_train: subset of output for training 
        Y_test: subset of output for testing
        size_train: size of the training sample
        
    """
    num_observation = np.shape(data)[0]
    size_train = int(num_observation*ratio_train)
    X_train = data[:size_train,:-cut]
    X_test = data[size_train:,:-cut]
    Y_train = np.reshape(data[:size_train,-cut:],(size_train,cut))
    Y_test = np.reshape(data[size_train:,-cut:],(num_observation-size_train,cut))
    return X_train , X_test , Y_train , Y_test ,size_train

def data_preparation_approach_1(data, ratio_train=0.8):
    """
    Prepare the dataset before training the neural network with the first approach
    
    Args:
        data: Sample of data
        ratio_train (double): ratio of training sample
        
    Results:
        X_train: subset of input for training 
        X_test: subset of input for testing 
        Y_train: subset of output for training 
        Y_test: subset of output for testing
        point_inter_test: point intersection with one of the edges
        true_angle_test: angle of the orientation of the discontinuities
        
    """
    num_observation = np.shape(data)[0]
    random_permuted_indices = np.random.permutation(num_observation)
    point_inter = data[random_permuted_indices,1:3]
    data_nn = data[random_permuted_indices,3:-1]
    true_angle = data[random_permuted_indices,-1]
    new_data = data_normalization(data_nn,cut=2)
    X_train , X_test , Y_train , Y_test, size_train = data_split( new_data, ratio_train = ratio_train,cut=2)
    point_inter_test = point_inter[size_train:]
    true_angle_test = true_angle[size_train:]
    return  X_train, X_test, Y_train, Y_test, point_inter_test, true_angle_test

def data_preparation_approach_2(data, ratio_train=0.8):
    """
    Prepare the dataset before training the neural network with the second approach
    
    Args:
        data: Sample of data
        ratio_train (double): ratio of training sample
        
    Results:
        X_train: subset of input for training 
        X_test: subset of input for testing 
        Y_train: subset of output for training 
        Y_test: subset of output for testing
        point_inter_test: point intersection with one of the edges
        true_angle_test: angle of the orientation of the discontinuities
        
    """
    num_observation = np.shape(data)[0]
    random_permuted_indices = np.random.permutation(num_observation)
    point_inter = data[random_permuted_indices,1:3]
    data_nn = data[random_permuted_indices,3:-1]
    true_angle = data[random_permuted_indices,-1]
    vecteur_norm = np.linalg.norm(data_nn[:,-2:],axis=1)
    angle = np.arctan2(data_nn[:,-2] ,data_nn[:,-1])
    data_nn[:,-2] = np.cos(angle)
    data_nn[:,-1] = np.sin(angle)
    data_nn = np.c_[data_nn ,vecteur_norm ]
    new_data = data_normalization(data_nn , cut = 3)
    X_train , X_test , Y_train , Y_test, size_train = data_split( new_data, ratio_train = ratio_train,cut=3)
    point_inter_test = point_inter[size_train:]
    true_angle_test = true_angle[size_train:]
    return  X_train, X_test, Y_train, Y_test, point_inter_test, true_angle_test
    
def data_normalization(data,cut=2):
    """
    Normalize the data between -1 and 1, for each observation
    
    Args:
        data: Samples of data
        cut (int): where to cut
        
    Returns:
        New_data: the normalize data
        
    """
    nb_col = np.shape(data[:,:-cut])[1]
    max_row = np.repeat(np.max( np.abs(data[:,:-cut]),axis = 1)[:,np.newaxis],nb_col,axis=1)
    New_data = np.copy(data)
    New_data[:,:-cut] /= max_row
    return New_data

def data_preparation(data, ratio_train=0.8):
    """
    Prepare the data before training the neural network 
    
    Args:
        data: Sample of data
        modifie_angle (bool): select true to transform the data
        ratio_train (double): ratio of training sample
        
    Returns:
        X_train: subset of input for training 
        X_test: subset of input for testing 
        Y_train: subset of output for training 
        Y_test: subset of output for testing
        indice_neg: indice where target are in [-pi, 0]
    
    """
    new_data = data_normalization(data)
    X_train , X_test , Y_train , Y_test,permuted_indices,size_train = data_split( new_data, ratio_train = ratio_train)
    permuted_indice_train = permuted_indices[:size_train]
    permuted_indice_test = permuted_indices[size_train:]
    return X_train , X_test , Y_train , Y_test, permuted_indice_train , permuted_indice_test

################### B. Training and testing of the neural network ####################
def compute_loss_modify(y_pred , y_true):
    check = np.sign( np.pi/2 - y_pred )*np.pi
    val1 = (y_pred-y_true)**2
    val2 = (y_true -(y_pred+check))**2
    return np.mean(np.minimum(val1,val2))

def best_angle(y_pred , y_true):
    y_pred_new = np.copy(y_pred)
    for ind_y,y_p in enumerate(y_pred):
        if y_p < np.pi/2 :
            if np.abs(y_p-y_true[ind_y]) > np.abs(y_true[ind_y] -(y_p+np.pi)):
                y_pred_new[ind_y]= y_p + np.pi
        else:
            if np.abs(y_p-y_true[ind_y]) > np.abs(y_true[ind_y] -(y_p-np.pi)):
                y_pred_new[ind_y]= y_p - np.pi
    return y_pred_new

def diagnostic_angle(y_pred , y_true ):
    
    y_pred_new = best_angle(y_pred,y_true)
                
    #check = np.sign( np.pi/2 - y_pred )*np.pi
    #tmp = np.c_[(y_pred-y_true)**2,(y_true -(y_pred+check))**2]
    #ind = np.argmin(tmp,axis= 1)
    #y_pred_new = y_pred+ind*check
    
    # radian to degree
    y_pred_degree = (180/np.pi)*y_pred_new
    y_true_degree = (180/np.pi)*y_true
    
    # compute some statistics
    
    error = np.abs(y_pred_degree-y_true_degree)
    mean = np.mean(error)
    std = np.std(error)
    max_ = np.max(error)
    min_ = np.min(error)
    median = np.median(error)
    nbr_value = error.shape[0]
    
    
    # the number of accuracy in %
  
    accuracy_1    = round(((np.where(error < 1)[0].shape[0])/nbr_value)*100,2)
    accuracy_5    = round(((np.where(error < 5)[0].shape[0])/nbr_value)*100,2)
    accuracy_10   = round(((np.where(error < 10)[0].shape[0])/nbr_value)*100,2)
    
    print("========================Summary angle=============================")
    print("all value are in degree ")
    print(" the error mean is : ",mean)
    print(" the error std is : ",std)
    print(" the median is : ",median)
    print(" the min is : ",min_)
    print(" the max is : ",max_)
    print(" the accuracy up to 1    degree is : ",accuracy_1," %")
    print(" the accuracy up to 5    degree is : ",accuracy_5," %")
    print(" the accuracy up to 10    degree is : ",accuracy_10," %")
    
    stock_value = [mean , std ,median, min_, max_, accuracy_1, accuracy_5, accuracy_10]
    
    return stock_value

def diagnostic_distance(y_pred , y_true ):
    
    # compute some statistics
    mse = np.mean((y_pred-y_true)**2)
    error = np.abs(y_pred-y_true)
    mean = np.mean(error)
    std = np.std(error)
    max_ = np.max(error)
    min_ = np.min(error)
    median = np.median(error)
    
    print("========================Summary distance=========================")
    print("all value are in degree ")
    print(" the mse is : ",mse )
    print(" the error mean is : ",mean)
    print(" the error std is : ",std)
    print(" the median is : ",median)
    print(" the min is : ",min_)
    print(" the max is : ",max_)
    
    stock_value = [mse ,mean , std ,median,min_ ,max_]
    
    return stock_value
    

def model_training(model,X_train,X_test,Y_train,Y_test, patience_ = 50 ,batch_size_ = 1000 , plot=True , approach = 1 ):
    """Train the different models and return the MSE values"""
    early_stop = EarlyStopping(monitor ="val_loss", patience = patience_ , verbose = 1)
    history = model.fit(X_train, Y_train, validation_split=0.2, epochs=2000,batch_size = batch_size_ ,callbacks=[early_stop] , verbose=0)
    train_mse = model.evaluate(X_train, Y_train, verbose=0)
    test_mse = model.evaluate(X_test, Y_test, verbose=0)
    
    # angle check 
    Y_pred = model.predict(X_test , verbose = 0 )
    true_angle = np.arctan2(Y_test[:,0] ,Y_test[:,1])
    pred_angle = np.arctan2(Y_pred[:,0 ] ,Y_pred[:,1] )
    results_angle = diagnostic_angle(pred_angle , true_angle )
    mse_angle = np.mean((true_angle - pred_angle)**2)
    mse_2_angle = compute_loss_modify(pred_angle , true_angle)
    print('MSE angle test: %.3f' % (mse_angle))
    print('MSE 2 angle test: %.3f' % (mse_2_angle))
    
    # distance check
    
    if approach == 1 :
        dist_pred = np.linalg.norm(Y_pred,axis=1)
        dist_true = np.linalg.norm(Y_test,axis=1)
    else :
        dist_pred = Y_pred[:,-1]
        dist_true = Y_test[:,-1]
        
    results_dist = diagnostic_distance(dist_pred, dist_true )
    
    print('MSE Train: %.3f | MSE Test: %.3f' % (train_mse, test_mse))
    if plot :
        # plot loss during training
        plt.title('Loss / Mean Squared Error')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
    return train_mse,test_mse , results_angle ,results_dist


def prediction_on_new_grid(data_path,output_path,model,vizu=False ,normalized=True):
    
    print("loading data ... ")
    data_test = np.loadtxt(data_path, delimiter=',')
    print("loading succed")
    
    n_row = np.shape(data_test)[0]
    Y_test = np.reshape(data_test[:,-1],(n_row,1))
    
    if normalized :
        nb_col = np.shape(data_test[:,1:-1])[1]
        max_row = np.repeat(np.max( np.abs(data_test[:,1:-1]),axis = 1)[:,np.newaxis],nb_col,axis=1)      
        X_test = data_test[:,1:-1]/max_row
    else :
         X_test = data_test[:,1:-1]
            
    Y_hat =  model.predict(X_test, verbose=0)
    
    print("saving prediction ...")
    path = os.getcwd()
    np.savetxt(path+output_path,Y_hat,delimiter=',')
    print("saving succed")
    


def build_model_40(summary = False,approach = 1):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(40 ,input_shape=(40,)))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(40))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(40))      
    model.add(Activation('sigmoid'))
    model.add(Dense(approach+1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

def build_model_84(summary = False,approach = 1):
    
   #architecture
    model = Sequential()
    
    model.add(Dense(84 ,input_shape=(84,)))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(84))        
    model.add(Activation('sigmoid'))
              
    model.add(Dense(84))      
    model.add(Activation('sigmoid'))
    model.add(Dense(approach+1))
              
    if summary:
        print(model.summary())
    
    #optimiser 
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model
