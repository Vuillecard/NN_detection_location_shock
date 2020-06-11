# Detecting discontinuity orientation and location on two-dimensional grids using neural networks
In a recent paper *Detecting troubled-cells on two-dimensional unstructured grids using a neural network* by Deep Ray and Jan S. Hesthaven, a neural network is used to detect discontinuities in the numerical solutions of two-dimensional conservation laws.  Based on this troubled-cells detector, the present work aims to construct a neural network to predict the orientation and the location of the shock in each troubled-cell.

## Technical detail :

>The code uses a keras implementation using tensorflow backend.
>All the notebooks are already run.
>The paths that are used in the code have to be slightly modified.
>The Data used in the project are too large (4.5 G) to be uploaded on Github. 
>Thus the data can be either recompute using the code in Matlab, 
>or it can be downloaded using this Wetransfer link:  https://we.tl/t-Wcw88iHozl

## Folder structure : 

- ***\Python code*** 

	- ***\Orientation detection***
	Contains folders and notebooks of the code used to compute and collect the results that are shown in the report.

        - ***\data_prediction :***
        It contains circular prediction in .csv file 
        - ***\Experiment :***
        Contains notebook that has been used to try and test new method.
        - ****\Results :****
        Contains results in csv file that is shown in the report.
        - ``circular_prediction.ipynb`` : Notebook that contains the prediction of a circular discontinuity for mesh C and few orders.
        -``data_exploration.ipynb`` : Notebook that describes the dataset.
        -`` Mesh dependency.ipynb`` : Notebook that contains the results of the mesh dependency and the robustness of the neural network.
        -``Model selection`` :  Notebook that contains the results of the model selection of different architecture of neural networks.
        -``Results_approach_`` :  Notebook that contains the results of different approaches to train the neural networks.
        -''function.py'' : contains all the functions that are used in the notebook.
        -''models.py`` : contains all the neural network architecture.
        
    - ***\Location detection :***
	Contains folders and notebooks of the code used to compute and collect the results that are shown in the report.

        - ***\data_prediction :***
        It contains circular prediction in .csv file 
        - ****\Results :****
        Contains results in csv file that is shown in the report.
        - ``circular_prediction.ipynb`` : Notebook that contains the prediction of a circular discontinuity for mesh C and few orders.
        -``data_exploration.ipynb`` : Notebook that describes the dataset.
        -`` Mesh dependency.ipynb`` : Notebook that contains the results of the mesh dependency and the robustness of the neural network.
        -``Approach_ .ipynb`` :  Notebook that contains the results of different approaches to train the neural networks.
        -``method_experiment.ipynb`` : Notebook that contains tests and visualisation some approaches.
        -''function.py'' : contains all the functions that are used in the notebook.
        -''models.py`` : contains all the neural network architecture.

- ***\Matlab code***
	Contains all functions needed to generate the data and visualise the prediction of the neural network. It also contains ``Matlab code description.txt`` that describes all the functions. 

- ***\Meshes***
Contains all the meshes used in the report. 

## Further work
Future work can be done. One should test the trained networks in a concrete, time-dependent problem. Given the good generalisation and robustness properties we showed, we expect the networks to be able to capture well both the direction and the location of discontinuities independently of the underlying problem. We could also investigate the computational cost of the online application of the neural network. As it mainly consists of matrix-vector multiplications, we expect it to be quite low. Moreover the neural network can be trained with noisy data and data generated from numerical simulations, we could expect better results in real applications.





