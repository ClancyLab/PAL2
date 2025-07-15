================
Running closed-loop framework
================

1. Configure the attributes in the ``surrogate_models_input.py`` file, save the file.
2. Open **Anaconda Prompt** and navigate to the directory where you've cloned the repository, i.e., the ``run_folder`` attribute in the input file..
3. Make sure that the **PAL_env** environment is activated
4. Running the code:

    - To perform Bayesian Optimization, run the following jupyter notebook:
 
     .. code-block:: bash
	
      acquitision_function_bo-ClosedLoop.ipynb

The code for the closed-loop PAL 2.0 framework can be found at https://github.com/ClancyLab/MPEA_PAL2
