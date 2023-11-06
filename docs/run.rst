================
Running PAL 2.0
================

1. After configuring the attributes in the ``user_configuration.yml`` file, save the file.
2. Open **Anaconda Prompt** and navigate to the directory where you've cloned the repository, i.e., the ``run_folder`` attribute in the YAML file.
3. Make sure that the **PAL_env** environment is activated
4. Running the Snakefile:

    - To perform Bayesian Optimization, run the following command:
 
     .. code-block:: bash

      snakemake --cores 1 perform_bo


    - To perform Predictive Inference, run the following command:
 
     .. code-block:: bash

      snakemake --cores 1 perform_pi

A text file ``success.txt`` would be created in the same directory if the task is performed successfully.
