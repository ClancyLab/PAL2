======================
Pipeline Configuration
======================

You can change attributes in the ``user_configuration.yml`` file to run **PAL 2.0** according to your custom choice.
The attributes for setting up the pipeline are given below:

---------------------
Directory Attributes
---------------------
- :bglb:`run_folder`
    * Address of the cloned repository on your system ending with a */*, preceeded by the ``home`` anchor
    * Example: C:/Users/MatDisc_ML/
    * This is a required value, there is no default value 
- :bglb:`output_folder`
    * List with ``home`` reference as first element and address of directory to save outputs of Bayesian Optimization ending with a */* as second element
    * Values = Any string literal for the second element
    * Default value = bo_output/ 

------------------
General Attributes
------------------
- :bglb:`test_size`
    * Fraction of the unobserved materials space that needs to be explored by Bayesian Optimization.
    * Example: If we have a total of 100 materials, then test_size = 0.9 implies, we will use 10 materials to train our surrogate models initially and then explore the remaining 90 based on Bayesian Optimization. 
    * Values = Floating point in (0,1) 
    * Default value = 0.9
- :bglb:`verbose`
    * Set to True if we want to print the progress of the code and Bayesian Optimization iterations without too much detail regarding the fitting process. It mainly prints out the outputs of the Bayesian Optimization iterations.
    * Values = :lime:`True` or :red:`False`
    * Default value = :lime:`True`
- :bglb:`deep_verbose`
    * Set to True if we want to print the progress of the code and Bayesian Optimization iterations with all details including training of the Gaussian Process models, prior means and Bayesian Optimization iteration outputs.
    * Values = :lime:`True` or :red:`False`
    * Default value = :red:`False`

----------------
Input Attributes
----------------
- :bglb:`dataset_folder`
    * Folder name suffix to save the Bayesian Optimization output for a given data. 
    * The standard format of the output folder name is: *dataset_folder* + *test_size* + *p_Run* + Run number 
    * Values = Any string literal
    * Default value = newDataset
- :bglb:`InputType`
    * The format in which your dataset is stored. 
    * Values = Gryffin, PerovAlloys, PALSearch, MPEA
    * Default value = Gryffin
    * Given below is a table which shows the various input types and their associated file extensions:

        ===========   ==================
        InputType      File Extension
        ===========   ==================
        Gryffin       .pkl
        PerovAlloys   .csv
        PALSearch     .xls, .xlsx
        MPEA          .xls, .xlsx
        ===========   ==================
- :bglb:`InputPath`
    * List with ``home`` reference as first element and address for where the dataset is saved ending with a */* as second element.
    * Values = Name of the directory where the dataset is stored
    * Default value = datasets/
- :bglb:`InputFile`
    * Name of the dataset file.
    * Values = Filename of the dataset being used
    * Default value = perovskites_GRYFFIN.pkl
- :bglb:`AddTargetNoise`
    * Set to True if we want to add a small Gaussian noise to the target property 
    * Values = :lime:`True` or :red:`False`
    * Default value = :red:`False`

----------------------------
Feature Selection Attributes
----------------------------
- :bglb:`test_size_fs`
    * Fraction of the data to be used to do feature selection. 
    * In the case of running Bayesian Optimization, this needs to be set the same as the *test_size* variable mentioned earlier.
    * Values = Floating point in (0,1) 
    * Default value = 0.1
- :bglb:`select_features_otherModels`
    * Set to True if we want to do feature selection of input descriptors for all models other than Gaussian Process - Neural Network model.
    * Values = :lime:`True` or :red:`False`
    * Default value = :lime:`True`
- :bglb:`select_features_NN`
    * Set to true if we want to do feature selection of input descriptors for the Gaussian Process - Neural Network model.
    * Values = :lime:`True` or :red:`False`
    * Default value = :lime:`True`

------------------------------------
Surrogate Models Training Attributes
------------------------------------
- :bglb:`train_NN`
    * Set to True if we want to train the Neural Network model initial before using the Neural Network as a prior mean to fit the Gaussian Process model,
    * Values = :lime:`True` or :red:`False`
    * Default value = :lime:`True`
- :bglb:`saveModel_NN`
    * Set to true if we want to save the Neural Network model in a file after fitting. 
    * This has to be set to True if we are training the Neural Network model with the given initial data for the first time. 
    * Values = :lime:`True` or :red:`False`
    * Default value = :lime:`True`
- :bglb:`train_GP`
    * Set to True if we want to train the Gaussian Process models
    * Values = :lime:`True` or :red:`False`
    * Default value = :lime:`True`
- :bglb:`predict_NN`
    * Set to True if we want to use the Neural Network model to do predictions
    * Values = :lime:`True` or :red:`False`
    * Default value = :red:`False`