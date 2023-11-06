====================================
Bayesian Optimization Configuration
====================================

Given below are the attributes needed for performing Bayesian Optimization:

--------------------------------
Bayesian Optimization Attributes
--------------------------------
- :bglb:`n_trials`
    * Number of repeats of the Bayesian Optimization process to gain convergence information of the performance of different models. 
    * Every trial is set to a different random seed which results in different sets of initial data used to initiate training of surrogate models before starting off the Bayesian Optimization iterations.
    * Values = Integer value greater than 0
    * Default value = 50
- :bglb:`n_update`
    * Number of Bayesian Optimization iterations at which we want to update our surrogate models. 
    * Values = Integer value greater than 1
    * Default value = 10
- :bglb:`random_seed`
    * Choice for choosing the random seed to initialize the initial training dataset.
    * *iteration* for trial number or *time* for CPU clock time. 
    * Values = iteration, time
    * Default value = iteration

------------------------------
GP Model Selection Attributes
------------------------------
- :bglb:`GP_0_BO`
    * Set to True if we want to run Bayesian Optimization for a Gaussian Process surrogate with 0-prior mean
    * Values = :lime:`True` or :red:`False`
    * Default value = :lime:`True`
- :bglb:`GP_C_BO`
    * Set to True if we want to run Bayesian Optimization for a Gaussian Process surrogate with constant function prior mean
    * Values = :lime:`True` or :red:`False`
    * Default value = :lime:`True`
- :bglb:`GP_L_BO`
    * Set to True if we want to run Bayesian Optimization for a Gaussian Process surrogate with linear function prior mean
    * Values = :lime:`True` or :red:`False`
    * Default value = :lime:`True`
- :bglb:`GP_NN_BO`
    * Set to True if we want to run Bayesian Optimization for a Gaussian Process surrogate with Neural Network prior mean
    * Values = :lime:`True` or :red:`False`
    * Default value = :lime:`True`

-------------------
GP Model Attributes
-------------------
- :bglb:`kernel`
    * Choice of the kernel for the Gaussian Process models. 
    * Values = Matern, RBF
    * Default value = Matern

^^^^^^^^^^^^^^^
Learning rates
^^^^^^^^^^^^^^^
- :bglb:`learning_rate_gp0`
    * Learning rate for training the Gaussian Process with 0-prior mean model
    * Values = Floating point in (0,1)
    * Default value = 0.1
- :bglb:`learning_rate_gpC`
    * Learning rate for training the Gaussian Process with constant function prior mean model
    * Values = Floating point in (0,1)
    * Default value = 0.1
- :bglb:`learning_rate_gpL`
    * Learning rate for training the Gaussian Process with linear function prior mean model
    * Values = Floating point in (0,1)
    * Default value = 0.1
- :bglb:`learning_rate_gpNN`
    * Learning rate for training the Gaussian Process with Neural Network prior mean model
    * Values = Floating point in (0,1)
    * Default value = 0.01

^^^^^^^
Epochs
^^^^^^^
- :bglb:`epochs_gp0`
    * Number of training epochs for training the Gaussian Process with 0-prior mean model
    * Values = Integer value greater than 0
    * Default value = 100
- :bglb:`epochs_gpC`
    * Number of training epochs for training the Gaussian Process with constant function prior mean model
    * Values = Integer value greater than 0
    * Default value = 100
- :bglb:`epochs_gpL`
    * Number of training epochs for training the Gaussian Process with linear function prior mean model
    * Values = Integer value greater than 0
    * Default value = 100
- :bglb:`epochs_gpNN`
    * Number of training epochs for training the Gaussian Process with Neural Network prior mean model
    * Values = Integer value greater than 0
    * Default value = 500

--------------------------
Neural Network Attributes
--------------------------
- :bglb:`learning_rate`
    * Learning rate for training the Neural Network model
    * Values = Floating point in (0,1)
    * Default value = 1e-6
- :bglb:`batch_size_nn`
    * Minibatch size to split up the data per epoch during the Neural Network training
    * Values = Integer value greater than 0
    * Default value = 5
- :bglb:`epochs`
    * Number of training epochs for training the Neural Network model
    * Values = Integer value greater than 0
    * Default value = 3000
- :bglb:`l1`
    * L1 regularization parameter for Neural Network weights
    * Values = Floating point in (0,1)
    * Default value = 0.1
- :bglb:`l2`
    * L2 regularization parameter for Neural Network weights
    * Values = Floating point in (0,1)
    * Default value = 0.4
- :bglb:`num_nodes`
    * Number of nodes in the first hidden layer of the Neural Network
    * Values = Integer value greater than 0
    * Default value = 50
- :bglb:`saveModel_filename`
    * File name that is used to the save the Neural Network model to be used later when fitting the Gaussian Process Neural Network model.  
    * Values = Any string literal 
    * Default value = connor_90p_chanNNarch.pt
- :bglb:`saveModel_folder`
    * Folder name where the abovementioned *saveModel_filename* is stored
    * Values = Any string literal
    * Default value = NN_savedmodels_BO/

------------------------
Acquisition Attributes
------------------------
- :bglb:`standardize_data`
    * Set to True if we want to normalize/standardize the input dataset.
    * Values = :lime:`True` or :red:`False`
    * Default value = :lime:`True`
- :bglb:`save_output`
    * Set to true if we want to save the output of the acquisition function
    * This is equivalent to saving the output from the Bayesian Optimization iterations. 
    * Values = :lime:`True` or :red:`False`
    * Default value = :lime:`True`
- :bglb:`n_batch`
    * Parameter for training of the acquisition function
    * Values = Integer value greater than 0
    * Default value = 100
- :bglb:`num_restarts`
    * Parameter for training of the acquisition function
    * Values = Integer value greater than 0
    * Default value = 10
- :bglb:`raw_samples`
    * Parameter for training of the acquisition function
    * Values = Integer value greater than 0
    * Default value = 512
