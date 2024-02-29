==================
Installation Guide
==================

To run the **PAL 2.0** code on your system, follow the given steps:

1. Clone the `PAL 2.0 GitHub repository <https://github.com/ClancyLab/PAL2>`_ locally on your system. To do so, click the :bgcode:`<> Code ▼` button and you can find various methods to clone the repository on your system. 
2. Download and install `Anaconda <https://www.anaconda.com/download>`_ on your system, if it is not already installed.
3. Open **Anaconda Prompt** and navigate to the directory where you've cloned the repository in Step 1.
4. Run the following command:
 
   .. code-block:: bash

    conda env create -f PAL_env.yml

This will create a virtual environment called **PAL_env** that contains all the required dependencies to run *PAL 2.0* on your system, including Snakemake.

5. Activate the above environment by running the following command:

   .. code-block:: bash

       conda activate PAL_env
