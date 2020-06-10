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
    C. vizualisation

Note:
    The implementatoin of the neural networks use the Keras package:
        https://keras.io/api/
    Also it use tensorflow as backend
    The code also use the Numpy package :
        https://numpy.org/doc/1.18/reference/index.html

"""


####################### A. Data manipulation #################################

def modifie_negative_angle( data ,apply = True):
    """ 
    Convert the target from [-pi, pi] to [0, pi]
    
    Args:
        data: Samples of data 
        apply (bool): select true to transform the data
        
    Returns:
        indice where target are in [-pi, 0]
        
    """
    indice_neg = np.where(data[:,-1]<0)[0]
    n = data.shape[0]
    if apply :
        data[indice_neg,-1] += np.pi
    return indice_neg


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


def data_split(data, ratio_train=0.8):
    """ 
    Split the data into train and test set
    
    Args:
        data: Samples of data 
        ratio_train (double): ratio of training sample
        
    Returns:
        X_train: subset of input for training 
        X_test: subset of input for testing 
        Y_train: subset of output for training 
        Y_test: subset of output for testing 
        random_permuted_indices: the suffle indices
        size_train: size of the training sample
        
    """
    num_observation = np.shape(data)[0]
    size_train = int(num_observation*ratio_train)
    random_permuted_indices = np.random.permutation(num_observation)
    data_suffle = data[random_permuted_indices]
    X_train = data_suffle[:size_train,:-1]
    X_test = data_suffle[size_train:,:-1]
    Y_train = np.reshape(data_suffle[:size_train,-1],(size_train,1))
    Y_test = np.reshape(data_suffle[size_train:,-1],(num_observation-size_train,1))
    return X_train , X_test , Y_train , Y_test , random_permuted_indices,size_train


def data_normalization(data):
    """
    Normalize the data between -1 and 1, for each observation
    
    Args:
        data: Samples of data
        
    Returns:
        New_data: the normalize data
        
    """
    nb_col = np.shape(data[:,:-1])[1]
    max_row = np.repeat(np.max( np.abs(data[:,:-1]),axis = 1)[:,np.newaxis],nb_col,axis=1)
    New_data = np.copy(data)
    # divide each rows by their absolute maximum 
    New_data[:,:-1] /= max_row
    return New_data


def data_preparation(data,modifie_angle, ratio_train=0.8):
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
    indice_neg = modifie_negative_angle(new_data,modifie_angle)
    X_train , X_test , Y_train , Y_test,permuted_indices,size_train = data_split( new_data, ratio_train = ratio_train)
    
    return X_train , X_test , Y_train , Y_test , indice_neg


#################### B. Training and testing of the neural network ####################

def compute_loss_modify (y_pred , y_true):
    """
    New loss definition that take orientation into account 
    
    Args:
        y_pred: vector of predicted target
        y_true: vector of true target 
        
    Returns: 
        the loss value
    
    """
    check = np.sign( np.pi/2 - y_pred )*np.pi
    val1 = (y_pred-y_true)**2
    val2 = (y_true -(y_pred+check))**2
    return np.mean(np.minimum(val1,val2))

def best_angle(y_pred , y_true):
    """
    Modifie the prediction by pi in order to match the orientation of the true target 
    
    Args:
        y_pred: vector of predicted target
        y_true: vector of true target 
        
    Returns:
        y_pred_new: vector of prediction that have the same orientation as the true target 
        
    """
    y_pred_new = np.copy(y_pred)
    for ind_y,y_p in enumerate(y_pred):
        if y_p < np.pi/2 :
            if np.abs(y_p-y_true[ind_y]) > np.abs(y_true[ind_y] -(y_p+np.pi)):
                y_pred_new[ind_y]= y_p + np.pi
        else:
            if np.abs(y_p-y_true[ind_y]) > np.abs(y_true[ind_y] -(y_p-np.pi)):
                y_pred_new[ind_y]= y_p - np.pi
    return y_pred_new

def diagnostic(y_pred , y_true, show_summary , best_angle_ ):
    """
    Give statistic results of the prediction
    
    Args:
        y_pred: vector of predicted target
        y_true: vector of true target 
        show_summary (bool): True to print the results
        best_angle_ (bool): true to modifie the prediction in order to have the best orientation 
        
    Returns:
        stock_value: list of statistic results of the abolute error (0->mean, 1->srt, 2->std, 3->max, 4->min, 5->median, 6->1 degree accuracy, 7->5 degree accuracy, 8->10 degree accuracy)
    """
    
    if best_angle_ :
        y_pred_new = best_angle(y_pred,y_true)
    else :
        y_pred_new = y_pred
                
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
    
    print("========================Summary=============================")
    print("all value are in degree ")
    print(" the error mean is : ",mean)
    print(" the error std is : ",std)
    print(" the median is : ",median)
    print(" the max is : ",max_)
    print(" the min is : ",min_)
    print(" the accuracy up to 1    degree is : ",accuracy_1," %")
    print(" the accuracy up to 5    degree is : ",accuracy_5," %")
    print(" the accuracy up to 10    degree is : ",accuracy_10," %")
    
    stock_value = [mean , std ,max_,min_,median,accuracy_1,accuracy_5,accuracy_10]
    
    return stock_value
    
def model_training(model,X_train,X_test,Y_train,Y_test, patience_ = 100 ,batch_size_ = 32 ,modifie_loss= False ,show_summary=True, plot=True ):
    """
    Train the neural network
    
    Args:
        model: Architecture of the neural network
        X_train: subset of input for training 
        X_test: subset of input for testing 
        Y_train: subset of output for training 
        Y_test: subset of output for testing
        patience_ (int): Patience parameter for the early stopping
        batch_size_ (int): Batch use to train in stochastic mode 
        modifie_loss (bool): true to evoluate the model with a new loss 
        show_summary (bool): true to show the diagnostic of the results 
        plot (bool): true to plot the convergence of the training and validation set 
        
    Returns:
        train_mse: evalutaion of the model on the training set
        test_mse: evalutation of the model on the test set 
        resutls: list of statistics of the prediction of the prediction on the test set
    
    """
    
    early_stop = EarlyStopping(monitor ="val_loss", patience = patience_ , verbose = 1)
    history = model.fit(X_train, Y_train, validation_split=0.2, epochs=2000, batch_size = batch_size_ ,callbacks=[early_stop] , verbose=0)
    train_mse = model.evaluate(X_train, Y_train, verbose=0)
    test_mse = model.evaluate(X_test, Y_test, verbose=0)
    y_pred = model.predict(X_test,verbose=0)
    if modifie_loss :
        
        loss = compute_loss_modify(y_pred,Y_test)
        print('MSE modifie Test: %.3f' % (loss))
        
    results = diagnostic(y_pred , Y_test, show_summary , modifie_loss)
    print('MSE Train: %.3f | MSE Test: %.3f ' % (train_mse, test_mse))
    
    if plot :
        # plot loss during training
        plt.title('Loss / Mean Squared Error')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
    return train_mse , test_mse, results 

def training_NN(architecture,X_train, X_test, Y_train, Y_test, name = None, patience_ = 100, batch = 32,modifie_loss = False):
    """
    Create and train the neural network given a specific architecture
    
    Args:
        architecture: Architecture of the neural network
        X_train: subset of input for training 
        X_test: subset of input for testing 
        Y_train: subset of output for training 
        Y_test: subset of output for testing
        name (string): name to save the model 
        patience_ (int): Patience parameter for the early stopping
        batch_size_ (int): Batch use to train in stochastic mode 
        modifie_loss (bool): true to evoluate the model with a new loss
        
    Returns:
        model: neural network after the training
        train_mse: evalutaion of the model on the training set
        test_mse: evalutation of the model on the test set 
        resutls: list of statistics of the prediction of the prediction on the test set
    """
    print("===================NN architecture==========================")
    model = architecture()
    print("=======================training=============================")
    t1_start = perf_counter()  
    train_mse,test_mse,results = model_training(model, X_train, X_test, Y_train, Y_test, patience_ ,batch,modifie_loss)
    #run your code
    t1_stop = perf_counter() 
    print("Training time is :",format((t1_stop-t1_start)/60,'.2f') ," minute")
    if name is not None:
        print("======================Saving model==========================")
        model.save(name)
        print("Saved model to disk")
    return model , train_mse, test_mse , results


def prediction_on_new_grid(data_path,output_path,model,vizu=False ,normalized=True,modifie_loss = False):
    """
    Test a model on a new dataset
    
    Args:
        data_path: path where to find the dataset
        output_path: path where to store the prediction of the prediction 
        model: the neural networks Keras object
        vizu (bool): true to plot some prediction in a reference tringle 
        normalized (bool): true to normalize the data before the prediction
        modifie_loss: True to also compute the modifie loss
    
    Returns:
        results: List of some statistic of the prediction
    """
    # load the dataset
    print("loading data ... ")
    data_test = np.loadtxt(data_path, delimiter=',')
    print("loading succed")
    
    # Prepare the dataset before prediction 
    indice_neg_test = modifie_negative_angle(data_test)
    n_row = np.shape(data_test)[0]
    Y_test = np.reshape(data_test[:,-1],(n_row,1))
    
    if normalized :
        nb_col = np.shape(data_test[:,1:-1])[1]
        max_row = np.repeat(np.max( np.abs(data_test[:,1:-1]),axis = 1)[:,np.newaxis],nb_col,axis=1)      
        X_test = data_test[:,1:-1]/max_row
    else :
         X_test = data_test[:,1:-1]
    
    # Prediction 
    Y_hat =  model.predict(X_test, verbose=0)
    test_mse = model.evaluate(X_test,Y_test, verbose=0)
    test_mse_modifie = compute_loss_modify(Y_hat,Y_test)
    print("on a new mesh the loss is :",format(test_mse,'.4f'))
    print("on a new mesh the modifie mse is :",format(test_mse_modifie,'.4f'))
    
    if modifie_loss :
        y_pred_new = best_angle(Y_hat , Y_test)
    else :
        y_pred_new = Y_hat
    
    # compute some statistic of the prediction 
    results = diagnostic(Y_hat , Y_test, True , modifie_loss)
    
    if vizu :
        angle_visualization(model,X_test,Y_test,min(10,n_row))
        
    # substract pi to the negative angle after the prediction:
    y_pred_new[indice_neg_test] -= np.pi
    
    # save the prediction 
    print("saving prediction ...")
    path = os.getcwd()
    np.savetxt(path+output_path,y_pred_new,delimiter=',')
    print("saving succed")
    
    return results  

def small_test(mesh , discont , model , modifie_loss ):
    """
    Test of 4 different orientation on a specific mesh
    
    Args:
        mesh: select the types of mesh
        discont: type of discontinuity R or 0_1
        model: Neural network keras object
    
    Results:
        res: list of stastistic of the prediction for each 4 different orientation 
    """
    angles = ["_3_all_30.csv","_3_all_60.csv","_3_all_90.csv","_3_all_circle.csv"]
    data_test =["\\data_test_set_p_3_all_30.csv",
                "\\data_test_set_p_3_all_60.csv",
                "\\data_test_set_p_3_all_90.csv",
                "\\data_test_set_p_3_all_circle.csv"]
    root = "C:\\Users\\pierr\\Desktop\\Projet de semestre I\\data\\test_set\\"
    
    root_path = "pred_data_set_"
    res = []
    for ind , angle in enumerate(angles):
        data_path = root + discont +"\\"+mesh+data_test[ind]
        output_path = root_path +mesh+ angle
        res.append(prediction_on_new_grid(data_path,output_path,model,modifie_loss = modifie_loss))
        
    return res
        
    
########################### C. vizualisation #############################
    
def angle_visualization(model , X_test ,Y_test, nbr_of_plot = 10):
    """
    Visualization of the predicted and true orientaion in the test set in a reference triangle
    
    Args:
        model: neural network Keras object
        X_test: subset of input for testing 
        Y_test: subset of output for testing
        nbr_of_plot: number of prediction to plot
        
    Results:
        the plots
    
    """
    x = np.linspace(-1,1,1001)
    Y_hat =  model.predict(X_test, verbose=0)
    coeff_true = np.tan(Y_test)
    coeff_hat = np.tan(Y_hat)
    pts = np.array([[-1,-1],[-1,1],[1,-1]])
    for i in range(nbr_of_plot):
        plt.figure()
        triangle = plt.Polygon(pts,fill=False)
        plt.gca().add_patch(triangle)
        plt.arrow(-0.5,0,np.cos(Y_test[i,0]),np.sin(Y_test[i,0]),head_width=0.05, head_length=0.1,color='r', label =' true ')
        plt.arrow(-0.5,0,np.cos(Y_hat[i,0]),np.sin(Y_hat[i,0]),head_width=0.05, head_length=0.1,color='b', label = 'prediction')
        plt.xlim(-1.5,1.7)
        plt.ylim(-1.5,1.5)
        plt.title(" Angle visualization red is true ")
        plt.show()

