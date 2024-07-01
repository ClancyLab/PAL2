#This file performs bayesian optimization
import numpy as np   
import pandas as pd
import os
import shutil
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

#Data reading specific modules
import json
import ruamel.yaml
import re

# Torch specific module imports
import torch
import gpytorch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.nn import functional as F

# botorch specific modules
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel

# Plotting libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
#%matplotlib inline
#%load_ext autoreload
#%autoreload 2

# Tick parameters
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1

plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 15

# User defined python classes and files
import sys
sys.path.insert(0, './feature_engineering/')
sys.stdout = open(snakemake.log[0],'w')

import utils_dataset as utilsd
from botorch.optim import optimize_acqf, optimize_acqf_discrete

from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement

import time
import warnings

import surrogate_models

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class acquisition: # Rename this since it is the optimization framework.
    
    
    def optimize_acqf_and_get_observation(self, acq_func, X_test, Y_test):
        """Optimizes the acquisition function, and returns a new candidate"""
        # optimize
        candidates, _ = optimize_acqf_discrete(
            acq_function=acq_func,
            choices=X_test,
            q=1,
            max_batch_size=2048,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            unique=True
        )

        # observe new values
        new_x = candidates.detach()
        b = [1 if torch.all(X_test[i].eq(new_x)) else 0 for i in range(0,X_test.shape[0]) ]
        b = torch.tensor(b).to(torch.int)
        index = b.nonzero()[0][0]
        new_y = torch.reshape(Y_test[0,index],(1,1))

        X_test_new = X_test[torch.arange(0, X_test.shape[0]) != index, ...]
        Y_test_new = Y_test[..., torch.arange(0, Y_test.shape[1]) != index]

        return new_x, new_y, index, X_test_new, Y_test_new

    
    def bo_iteration(self, acq_func, X_train, X_test, Y_train, Y_test,scalerX_transform,scalerY_transform):

        best_observed=[]
        new_x_observed=[]
        new_y_observed=[]
        index_observed=[]
        
        # Doing this so that we can extract the index in the dataset for the point recommended by BO
        X_test_all = X_test
        Y_test_all = Y_test
            
        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, self.n_batch + 1):
            if ((iteration-1)%self.n_update==0):
                # fit the models every n_update iterations
                model_gp, likelihood_gp = self.Train_object.train_surrogate(X_train, Y_train, self.kernel)

            # optimize and get new observation using acquisition function
            new_x, new_y, index, X_test_new, Y_test_new = self.optimize_acqf_and_get_observation(acq_func, X_test, Y_test)

            # Update remaining choices tensor
            X_test = X_test_new
            Y_test = Y_test_new

            # Update training points
            X_train = torch.cat([X_train, new_x])
            Y_train = torch.cat([Y_train[0], new_y[0]])
            Y_train = torch.reshape(Y_train,(1,Y_train.shape[0]))
            
            # Get inverse transform for the normalized data and index of the recommended points in the dataset
            new_x_inv_transformed = scalerX_transform.inverse_transform(new_x[0].numpy().reshape(1, -1), copy=None)
            new_y_inv_transformed = scalerY_transform.inverse_transform(new_y[0].numpy().reshape(1, -1), copy=None)
            
            b = [1 if torch.all(X_test_all[i].eq(new_x)) else 0 for i in range(0,X_test_all.shape[0])]
            b = torch.tensor(b).to(torch.int)
            index = b.nonzero()[0][0]
            
            # update progress
            best_value_ei = Y_train.max()
            best_observed.append(best_value_ei)
            new_x_observed.append(new_x_inv_transformed[0])
            index_observed.append(index.numpy())
            new_y_observed.append(new_y_inv_transformed[0][0])

            acq_func = ExpectedImprovement(model=model_gp, best_f=best_value_ei, maximize=self.maximization)

            if self.verbose:
                print(
                    f"\nBatch {iteration:>2}: best_value = {best_value_ei:>4.2f}",end="",)

        return best_observed, new_x_observed, new_y_observed, index_observed
    
    
    def best_in_initial_data(self, X_train, X_test, Y_train, Y_test, scalerX_transform, scalerY_transform): # need to rename this function. It is finding the best in initial data but also calls the bo_iteration function
    
        best_observed = []
        new_x_observed=[]
        new_y_observed=[]
        index_observed=[]
        
        # Finding best value in initial data
        if self.maximization:
            best_observed_value = Y_train.max()
            optimal_solution = torch.cat([Y_train[0],Y_test[0]]).max()
        else:
            best_observed_value = Y_train.min()
            optimal_solution = torch.cat([Y_train[0],Y_test[0]]).min()

        # If optimal value is present in the initial dataset sample remove it  
        if (best_observed_value.eq(optimal_solution)) and self.maximization:
            print('Max in training set, removing it before training models.')
            optimal_position = torch.argmax(Y_train)

            # Add max value to test/exploration set
            X_add_toTest = torch.reshape(X_train[optimal_position,:],(1,X_train.shape[1]))
            X_test = torch.cat([X_test,X_add_toTest])
            Y_add_toTest = torch.reshape(optimal_solution,(1,1))      
            Y_test = torch.cat((Y_test,Y_add_toTest),1)

            # Remove max value from training set
            X_train = X_train[torch.arange(0, X_train.shape[0]) != optimal_position, ...]
            Y_train = Y_train[..., torch.arange(0, Y_train.shape[1]) != optimal_position]

            # Update best observed value
            best_observed_value = Y_train.max()

        elif (best_observed_value.eq(optimal_solution)) and not self.maximization:
            print('Min in training set, removing it before training models.')
            optimal_position = torch.argmin(Y_train)

            # Add min value to test/exploration set
            X_add_toTest = torch.reshape(X_train[optimal_position,:],(1,X_train.shape[1]))
            X_test = torch.cat([X_test,X_add_toTest])
            Y_add_toTest = torch.reshape(optimal_solution,(1,1))      
            Y_test = torch.cat((Y_test,Y_add_toTest),1)

            # Remove min value from training set
            X_train = X_train[torch.arange(0, X_train.shape[0]) != optimal_position, ...]
            Y_train = Y_train[..., torch.arange(0, Y_train.shape[1]) != optimal_position]

            # Update best observed value
            best_observed_value = Y_train.min()

        # Initialize likelihood and model
        likelihood_gp = gpytorch.likelihoods.GaussianLikelihood()
        model_gp = self.gp_model(X_train, Y_train, likelihood_gp, self.kernel, *self.additional_arguments)

        # Initializing acquisition function for the models
        AcqFunc = ExpectedImprovement(model=model_gp, best_f=best_observed_value, maximize=self.maximization)
        best_observed.append(best_observed_value)
        best_observed_bo, new_x_observed, new_y_observed, index_observed = self.bo_iteration(AcqFunc, X_train, X_test, 
                                                                                             Y_train, Y_test, scalerX_transform, scalerY_transform)
        best_observed.extend(best_observed_bo)
    
        return best_observed, new_x_observed, new_y_observed, index_observed

    
    def generate_csv(self, best_observed_all,x_observed_all, y_observed_all, index_observed_all):
        best_observed_df = pd.DataFrame((np.array(best_observed_all).T[:self.n_batch,:]), columns = list(range(1, self.n_trials + 1))).add_prefix('trial_')
        best_observed_df.to_csv(self.run_folder+self.model_name+'_best_target_observed_normalized.csv',index=False)
        with open(snakemake.output[0],'a') as f:
            f.write(self.run_folder+self.model_name+"_best_target_observed_normalized.csv\n")
        
        print(np.size(x_observed_all))
        # x_observed_df = pd.DataFrame((np.array(x_observed_all).T[:self.n_batch,:]), columns = list(range(1, self.n_trials + 1))).add_prefix('trial_')
        # x_observed_df.to_csv(self.run_folder+self.model_name+'_x_observed.csv',index=False)
        # with open(snakemake.output[0],'a') as f:
        #     f.write(self.run_folder+self.model_name+"_x_observed.csv\n")
            
        y_observed_df = pd.DataFrame((np.array(y_observed_all).T[:self.n_batch,:]), columns = list(range(1, self.n_trials + 1))).add_prefix('trial_')
        y_observed_df.to_csv(self.run_folder+self.model_name+'_y_observed.csv',index=False)
        with open(snakemake.output[0],'a') as f:
            f.write(self.run_folder+self.model_name+"_y_observed.csv\n")
            
        index_observed_df = pd.DataFrame((np.array(y_observed_all).T[:self.n_batch,:]), columns = list(range(1, self.n_trials + 1))).add_prefix('trial_')
        index_observed_df.to_csv(self.run_folder+self.model_name+'_index_observed.csv',index=False)
        with open(snakemake.output[0],'a') as f:
            f.write(self.run_folder+self.model_name+"_index_observed.csv\n")
        
        
    def data_generation(self):
        best_observed_all_trials = []
        x_observed_all_trials = []
        y_observed_all_trials = []
        index_observed_all_trials = []
        
        # Average over multiple trials
        for trial in range(1, self.n_trials + 1):
            t0 = time.monotonic()
            if self.random_seed == 'time':
                random_seed = int(t0)
            elif self.random_seed == 'iteration':
                random_seed = trial

            print(f"\n -------------------- Trial {trial:>2} of {self.n_trials} --------------------\n", end="")

            # Getting initial data and fitting models with initial data
            if self.standardize_data:
                if not self.additional_arguments:
                    X_train, X_test, Y_train, Y_test, Var_train, Var_test, scalerX_transform, scalerY_transform = utilsd.generate_training_data(random_seed,self.test_size,snakemake.input[0])
                else:
                    X_train, X_test, Y_train, Y_test, Var_train, Var_test, scalerX_transform, scalerY_transform = utilsd.generate_training_data_NN(random_seed,self.test_size,snakemake.input[0])
            else:
                if not self.additional_arguments:
                    X_train, X_test, Y_train, Y_test, Var_train, Var_test = utilsd.generate_training_data(random_seed,self.test_size,snakemake.input[0])
                else:
                    X_train, X_test, Y_train, Y_test, Var_train, Var_test = utilsd.generate_training_data_NN(random_seed,self.test_size,snakemake.input[0])

            if self.additional_arguments:
                self.Train_object.train_surrogate_NN(X_train, Y_train,self.saveModel_filename,self.num_nodes,
                                                self.learning_rate,self.batch_size_nn,self.epochs,self.l1,self.l2,
                                                 self.saveModel_NN)

            # Appending to common list of best observed values, with number of rows equal to number of trials
            best_observed_per_trial, x_observed_per_trial, y_observed_per_trial, index_observed_per_trial = self.best_in_initial_data(X_train, X_test, 
                                                                                                            Y_train, Y_test, 
                                                                                                            scalerX_transform, scalerY_transform)

            t1 = time.monotonic()
            best_observed_all_trials.append(best_observed_per_trial)
            x_observed_all_trials.append(x_observed_per_trial)
            y_observed_all_trials.append(y_observed_per_trial)
            index_observed_all_trials.append(index_observed_per_trial)
            
            print(f"\ntime = {t1-t0:>4.2f}.\n")

        if self.save_output:
            self.generate_csv(best_observed_all_trials, x_observed_all_trials, y_observed_all_trials, index_observed_all_trials)
            
        
    def model_decision(self,decisions):
        if decisions['GP_0']:
            self.gp_model = surrogate_models.ZeroGPModel
            self.model_name = 'gp_0'
            self.Train_object = surrogate_models.Train(self.model_name, self.learning_rate_gp0, self.epochs_GP0)
            self.Train_object.learning_rate = self.learning_rate_gp0
            self.Train_object.epochs = self.epochs_GP0
            self.data_generation()
        if decisions['GP_C']:
            self.gp_model=surrogate_models.ConstantGPModel
            self.model_name = 'gp_C'
            self.Train_object = surrogate_models.Train(self.model_name, self.learning_rate_gpC, self.epochs_GPC)
            self.Train_object.learning_rate = self.learning_rate_gpC
            self.Train_object.epochs = self.epochs_GPC
            self.data_generation()
        if decisions['GP_L']:
            self.gp_model=surrogate_models.LinearGPModel
            self.model_name = 'gp_L'
            self.Train_object = surrogate_models.Train(self.model_name, self.learning_rate_gpL, self.epochs_GPL)
            self.Train_object.learning_rate = self.learning_rate_gpL
            self.Train_object.epochs = self.epochs_GPL
            self.data_generation()
        if decisions['GP_NN']:
            self.gp_model=surrogate_models.NNGPModel
            self.model_name = 'gp_NN'
            self.additional_arguments.extend([self.saveModel_filename, self.num_nodes])
            self.Train_object = surrogate_models.Train(self.model_name, self.learning_rate_gpNN, self.epochs_GPNN,
                                      self.saveModel_filename, self.num_nodes)
            self.Train_object.learning_rate = self.learning_rate_gpNN
            self.Train_object.epochs = self.epochs_GPNN
            self.data_generation()
    
    
    def check(self,dictionary, key, category):
        # Initial conditions
        if key not in dictionary.keys() or dictionary[key] is None:
            return defaults[key]

        # Type-wise conditions

        if category=='A': #Boolean Attributes
            if isinstance(dictionary[key],bool):
                return dictionary[key]
            else:
                return defaults[key]

        if category=='B': #Float Attributes
            if isinstance(dictionary[key],float) and 0<dictionary[key]<1:
                return dictionary[key]
            else:
                return defaults[key]

        if category=='C': #Integer Attributes
            if isinstance(dictionary[key],int):
                return dictionary[key]
            else:
                return defaults[key]

        if category=='D': #String Attributes
            if key=='random_seed' and dictionary[key] in ['iteration','time']:
                return dictionary[key]
            else:
                return defaults[key]
            if key=='kernel' and dictionary[key] in ['Matern','RBF']:
                return dictionary[key]
            else:
                return defaults[key]
        
        
    def __init__(self, att):
        
        #Attributes needed for model_decision()
        self.GP_0_BO = self.check(att, 'GP_0_BO','A')
        self.GP_C_BO = self.check(att, 'GP_C_BO','A')
        self.GP_L_BO = self.check(att, 'GP_L_BO','A')
        self.GP_NN_BO = self.check(att, 'GP_NN_BO','A')
        
        self.additional_arguments = []
        
        #Attributes needed for model_decision()
        self.saveModel_filename = att['saveModel_filename']
        self.num_nodes = self.check(att,'num_nodes','C')
        
        #Attributes needed for data_generation()
        self.n_trials = self.check(att,'n_trials','C')
        self.random_seed = self.check(att,'random_seed','D')
        self.standardize_data = self.check(att,'standardize_data','A') 
        self.test_size = self.check(att,'test_size','B')
        self.save_output = self.check(att,'save_output','A') 
        
        #Attributes needed for best_in_initial_data()
        self.kernel = self.check(att,'kernel','D')
        
        #Attributes needed for bo_iteration()
        self.n_batch = self.check(att,'n_batch','C') 
        self.n_update = self.check(att,'n_update','C')
        self.verbose = self.check(att,'verbose','A')
        
        #Attributes needed for optimize_acqf_and_get_observation()
        self.batch_size_nn = self.check(att,'batch_size_nn','C') 
        self.num_restarts = self.check(att,'num_restarts','C') 
        self.raw_samples = self.check(att,'raw_samples','C') 
        self.maximization = self.check(att,'maximization','A')
        
        #Attributes needed for generate_csv()
        self.run_folder = att['run_folder']
        self.output_folder = att['output_folder']
        
        #Attributes needed for train_surrogate_NN() method of Train class
        self.learning_rate = self.check(att,'learning_rate','B')
        self.epochs = self.check(att,'epochs','C')
        self.l1 = self.check(att,'l1','B')
        self.l2 = self.check(att,'l2','B')
        self.saveModel_NN = self.check(att,'saveModel_NN','A')
        
        #Attributes needed for train_surrogate() method of Train class
        self.learning_rate_gp0 = self.check(att,'learning_rate_gp0','B')
        self.epochs_GP0 = self.check(att,'epochs_GP0','C')
        self.learning_rate_gpC = self.check(att,'learning_rate_gpC','B')
        self.epochs_GPC = self.check(att,'epochs_GPC','C')
        self.learning_rate_gpL = self.check(att,'learning_rate_gpL','B')
        self.epochs_GPL = self.check(att,'epochs_GPL','C')
        self.learning_rate_gpNN = self.check(att,'learning_rate_gpNN','B')
        self.epochs_GPNN = self.check(att,'epochs_GPNN','C')
        
        self.model_decision({'GP_0':self.GP_0_BO,'GP_C':self.GP_C_BO,'GP_L':self.GP_L_BO,'GP_NN':self.GP_NN_BO})
        
        
    def __repr__(self):
        return ""




# Comment for Maitreyee: Put in a file called pal2.py which will be the file being called in the Snakemake
#Reading default values
with open(snakemake.input[1],'r') as f:
    defaults=json.load(f)

#Reading user input values
with open(snakemake.input[0],'r') as f:
    user_data=ruamel.yaml.safe_load(f)
user_data['output_folder']=user_data['output_folder'][0]+user_data['output_folder'][1]
saveModel_folder=user_data['run_folder']+user_data['saveModel_filename'][0]
saveModel_filename=saveModel_folder+user_data['saveModel_filename'][1]

if not user_data['saveModel_NN']:
    #Checking if saveModel_filename exists or not
    #Throwing an error with appropriate message and terminating the program if it does not exist
    assert os.path.exists(saveModel_filename), f'saveModel_filename does not exist'
    

user_data['kernel']=user_data['kernel'][0]
user_data['random_seed']=user_data['random_seed'][0]

#Checking if run folder exists or not
#Throwing an error with appropriate message and terminating the program if it does not exist
assert os.path.exists(user_data['run_folder']), f'Run folder does not exist'

# Creating the folder within the output folder that will contain the saved outputs
out_folder = user_data['output_folder']+user_data['dataset_folder']+str(user_data['test_size'])+'p_Run'

try:
    new_num_run=max([int(re.findall(user_data['output_folder']+user_data['dataset_folder']+str(user_data['test_size'])+'p_Run(.+)',i.path)[0]) 
                     for i in os.scandir(user_data['output_folder']) if i.is_dir() and 
                     re.search(out_folder,i.path)])+1
except:
    new_num_run=1

out_folder += str(new_num_run) + '/'
# Create a new directory if it does not exist
isExist = os.path.exists(out_folder)
if not isExist:
    os.makedirs(out_folder)
    print("The new directory is created:\n"+out_folder)

with open(snakemake.output[0],'a') as f:
        f.write(out_folder+"\n")

saveModel_folder=out_folder+user_data['saveModel_filename'][0]
saveModel_filename=saveModel_folder+user_data['saveModel_filename'][1]
user_data['saveModel_filename'] = saveModel_filename
#Creating a folder for saving NN model if it does not exist
isExist = os.path.exists(saveModel_folder)
if not isExist:
    os.makedirs(saveModel_folder)
    print("The new directory is created:\n"+saveModel_folder)

# Copy input YAML configuration file to output folder
shutil.copy2(snakemake.input[0],out_folder)
# Copy surrogate model file to output folder
shutil.copy2('scripts/surrogate_models.py',out_folder)
user_data['output_folder']=out_folder

#THIS DOES EVERYTHING:
acquisition(user_data)

with open(snakemake.output[0],'a') as f:
        f.write(user_data['run_folder']+snakemake.log[0]+"\n")
