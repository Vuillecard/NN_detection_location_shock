# Detecting discontinuity orientation and location on two-dimensional grids using neural networks
In a recent paper *Detecting troubled-cells on two-dimensional unstructured grids using a neural network* by Deep Ray and Jan S. Hesthaven, a neural network is used to detect discontinuities in the numerical solutions of two-dimensional conservation laws.  Based on this troubled-cells detector, the present work aims to construct a neural network to predict the orientation and the location of the shock in each troubled-cell.

## Technical detail
The code uses a keras implementation using tensorflow backend.
All the notebook are alredy run.
The paths that are used in the code have to be slightly modifie.

## Further work
Future work can be done. One should test the trained networks in a concrete, time-dependent problem. Given the good generalization and robustness properties we showed, we expect the networks to be able to capture well both the direction and the location of discontinuities independently of the underlying problem. We could also investigate the computational cost of the online application of the neural network. As it mainly consists of matrix-vector multiplications, we expect it to be quite low. Moreover the neural network can be trained with noisy data and data generated from numerical simulations, we could expect better results in real applications.

## Folder structure 

### Python code 

#### Orientation detection and Discontinuity location
Contains folders and notebook of the code used to compute and collect the results that are shown in the report.

##### data
It contains all the data needed to train and test the neural network.
Note that discontinuity R means that the value of the discontinuity is taken in R, whereas dicontinuity_0_1 means that the value of the discontinuity is 0 and 1.
##### data_prediction
It contains some prediction
##### Experiment
Contains notebook that has been used to try and test new method.
##### Results
Contains results in csv file that is shown in the report.

### Matlab code
 - data_gen... are used to generate all the data using in the code can be found in the folder data
 - data_test... are used to generate all the test set
 - pred_plot... are used to plot the prediction in a given mesh 

### Meshes
Contains all the meshes used in the report. 





