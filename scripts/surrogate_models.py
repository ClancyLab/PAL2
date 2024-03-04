# This file costructs surrogate models for the input datasets
import numpy as np     
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

# botorch specific modules
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel

# Torch specific module imports
import torch
import gpytorch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.nn import functional as F

# Plotting libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

# User defined python classes and files
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../feature_engineering/')

import utils_dataset as utilsd
import input_class 
import code_verification as verification

np.random.seed(0)
torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n")


class NeuralNetwork(nn.Module): # Comment for Maitreyee: Add input flexibility to automatically create the NN. Create a separate NN.yaml file or add in the yaml file under NN parameters.
    def __init__(self, in_features, num_nodes):
        super(NeuralNetwork,self).__init__()

        self.in_features = in_features
        
        # layer_list = torch.nn.ModuleList()
        
        # self.layer1 = nn.Linear(in_features, num_nodes,bias=True)
        # self.layer2 = nn.Linear(num_nodes,1,bias=True)
        self.layer1 = nn.Linear(in_features, num_nodes,bias=True)
        self.layer2 = nn.Linear(num_nodes, 20,bias=True)
        self.layer3 = nn.Linear(20,1,bias=True)
        
    def forward(self, x):
        layer1_out = F.tanh(self.layer1(x))
        layer2_out = F.tanh(self.layer2(layer1_out))
        output = F.relu(self.layer3(layer2_out))
        # layer1_out = F.relu(self.layer1(x)) 
        # output = self.layer2(layer1_out)
        
        return output
    


class Train:
    def __init__(self,model_name, learning_rate, epochs, 
                 saveModel_filename=None, num_nodes=None,
                 verbose=True, deep_verbose=False):
        self.additional_arguments = []
        if model_name=='gp_0':
            self.gp_model = ZeroGPModel
        if model_name=='gp_C':
            self.gp_model = ConstantGPModel
        if model_name=='gp_L':
            self.gp_model = LinearGPModel
        if model_name=='gp_NN':
            self.gp_model = NNGPModel
            self.additional_arguments.extend([saveModel_filename, num_nodes])
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.verbose=verbose
        self.deep_verbose=deep_verbose
        print("\n"+model_name.upper()+' Model')
        if self.verbose:
            print('Starting training')
            
            

    def train_nn_loop(self, dataloader, model, loss_fn, optimizer,lambda1,lambda2):
        
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        train_loss = 0.0
        l1_regularization, l2_regularization = 0.0, 0.0
        
        for batch, sample_batched in enumerate(dataloader):
            # Compute prediction and loss
            X = sample_batched['in_features']
            y = sample_batched['labels']
            # var = sample_batched['variance']
            pred = model(X)
            train_loss += loss_fn(pred, y).item()
            pred_loss = loss_fn(pred, y)
            
            all_linear1_params = torch.cat([x.view(-1) for x in model.layer1.parameters()])
            all_linear2_params = torch.cat([x.view(-1) for x in model.layer2.parameters()])
            all_linear3_params = torch.cat([x.view(-1) for x in model.layer3.parameters()])
            l1_regularization = lambda1 * (torch.norm(all_linear1_params, 1)
                                        +  torch.norm(all_linear2_params, 1)
                                        +  torch.norm(all_linear3_params, 1))
            l2_regularization = lambda2 * (torch.norm(all_linear1_params, 2)
                                        +  torch.norm(all_linear2_params, 2)
                                        +  torch.norm(all_linear3_params, 2))

            # l1_regularization = lambda1 * (torch.norm(all_linear1_params, 1)+torch.norm(all_linear2_params, 1))
            # l2_regularization = lambda2 * (torch.norm(all_linear1_params, 2)+torch.norm(all_linear2_params, 2))

            loss = pred_loss + l1_regularization + l2_regularization 

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /=num_batches
        return train_loss
    
    
    
    def train_surrogate_NN(self,X_train,Y_train,saveModel_filename,num_nodes,
                           learning_rate,batch_size_nn,epochs,l1,l2,saveModel_NN):
    
        # NN Model, Loss and Optimizer
        model = NeuralNetwork(X_train.shape[1],num_nodes)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Dataloader for pytorch
        train_data = utilsd.InputDataset(X_train,Y_train)
        train_dataloader = DataLoader(train_data, batch_size=batch_size_nn, shuffle=True)

        train_loss = []
        for t in range(epochs):
            train_loss_epoch = self.train_nn_loop(train_dataloader, model, loss_fn, optimizer, l1, l2)
            train_loss.append(train_loss_epoch)
            if ((t+1)%100 == 0 and self.verbose):
                print(f"Epoch {t+1}---> training error: {train_loss_epoch:>7f}")

        if saveModel_NN:
            torch.save(model.state_dict(), saveModel_filename)

        return
    
    
    
    def train_surrogate(self, X_train, Y_train, kernel):

        mse = 0.0

        # Initialize likelihood and model
        likelihood_gp = gpytorch.likelihoods.GaussianLikelihood()
        model_gp = self.gp_model(X_train, Y_train, likelihood_gp, kernel, *self.additional_arguments)

        # Find optimal model hyperparameters
        model_gp.train()
        likelihood_gp.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model_gp.parameters(), lr=self.learning_rate)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_gp, model_gp)

        for i in range(self.epochs):
            optimizer.zero_grad()         # Zero gradients from previous iteration
            output = model_gp(X_train)  # Output from model
            loss = -mll(output, Y_train)  # Calc loss and backprop gradients 
            loss.backward()
            optimizer.step()

        return model_gp, likelihood_gp
    


    def predict_surrogates(model, likelihood, X):
    
        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_mean = model(X)
            observed_pred = likelihood(model(X))
            
        return observed_mean, observed_pred
    


    def initial_train_surrogates(self,X_train, X_test, Y_train, Y_test, Var_train, Var_test, saveModel_filename,**kwargs):

    
        # NN parameters
        learning_rate =  kwargs['learning_rate']
        batch_size = kwargs['batch_size']
        epochs = kwargs['epochs']

        l1 = kwargs['l1']
        l2 =  kwargs['l2']

        # NN Model, Loss and Optimizer
        model = NeuralNetwork(X_train.shape[1],kwargs['num_nodes'])
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        mse_gp0 = 0.0
        mse_gpcon = 0.0
        mse_gplinear = 0.0
        mse_gpnn = 0.0

        if (kwargs['train_NN']):
            # Dataloader for pytorch
            train_data = utilsd.InputDataset(X_train,Y_train,Var=Var_train)
            test_data = utilsd.InputDataset(X_test,Y_test,Var=Var_test)

            train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

            user_training = Train_NN()

            train_loss = []
            test_loss = []
            for t in range(epochs):
                train_loss_epoch = user_training.train_loop(train_dataloader, model, loss_fn, optimizer, l1, l2)
                test_loss_epoch = user_training.test_loop(test_dataloader, model, loss_fn)
                train_loss.append(train_loss_epoch)
                test_loss.append(test_loss_epoch)
                if ((t+1)%100 == 0 and self.verbose):
                    print(f"Epoch {t+1}---> training error: {train_loss_epoch:>7f}, val error: {test_loss_epoch:>7f}")

            if self.verbose:
                fig, ax = plt.subplots(figsize=(6,4))
                ax.plot(range(epochs),train_loss, label=f'Training Error,{train_loss_epoch:>7f}')
                ax.plot(range(epochs),test_loss, label=f'Validation Error,{test_loss_epoch:>7f}')
                ax.set_xlabel('Num. of epochs')
                ax.set_ylabel('MSE Loss')
                plt.legend()
            print("NN training Done!")

            if kwargs['saveModel_NN']:
                torch.save(model.state_dict(), saveModel_filename)

        if (kwargs['predict_NN']):
            mean_module = NeuralNetwork(X_train.shape[1],kwargs['num_nodes'])
            mean_module.load_state_dict(torch.load(saveModel_filename))
            mean_module.eval()

            output_nn = mean_module(X_train)

        if (kwargs['train_GP']):

            #--------------------------- GP-0 ---------------------------#
            # initialize likelihood and model
            print('GP-0 Model')
            training_iter = kwargs['epochs_GP0']
            likelihood_gp0 = gpytorch.likelihoods.GaussianLikelihood()
            model_gp0 = ZeroGPModel(X_train, Y_train, likelihood_gp0, kwargs['kernel'])

            # Find optimal model hyperparameters
            model_gp0.train()
            likelihood_gp0.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(model_gp0.parameters(), lr=kwargs['learning_rate_gp0'])  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_gp0, model_gp0)
            mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood_gp0, model_gp0)

            for i in range(training_iter):
                optimizer.zero_grad()        # Zero gradients from previous iteration
                output = model_gp0(X_train)  # Output from model
                loss = -mll(output, Y_train) # Calc loss and backprop gradients            
                loss.backward()
                if self.deep_verbose:
                    print('Iter %d/%d - Loss: %.3f  lengthscale: %.3f   noise: %.3f' % (
                        i + 1, training_iter, loss.item(),
                        model_gp0.covar_module.base_kernel.lengthscale.item(),
                        model_gp0.likelihood.noise.item()))            
                optimizer.step()        

            # Get into evaluation (predictive posterior) mode
            model_gp0.eval()
            likelihood_gp0.eval()

            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_mean = model_gp0(X_test)
                observed_pred = likelihood_gp0(model_gp0(X_test))
            num_points = Y_test.size()[1]
            mse_gp0 = 1.0/num_points*torch.sum(torch.square(Y_test - observed_mean.loc))


            #--------------------------- GP-Constant ---------------------------#
            # initialize likelihood and model
            print('GP-Constant Model')
            training_iter = kwargs['epochs_gpC']
            likelihood_gpC = gpytorch.likelihoods.GaussianLikelihood()
            model_gpC = ConstantGPModel(X_train, Y_train, likelihood_gpC, kwargs['kernel'])

            # Find optimal model hyperparameters
            model_gpC.train()
            likelihood_gpC.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(model_gpC.parameters(), lr=kwargs['learning_rate_gpC'])  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_gp0, model_gp0)
            mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood_gpC, model_gpC)

            for i in range(training_iter):
                optimizer.zero_grad()        # Zero gradients from previous iteration
                output = model_gpC(X_train)  # Output from model
                loss = -mll(output, Y_train) # Calc loss and backprop gradients            
                loss.backward()
                if self.deep_verbose:
                    print('Iter %d/%d - Loss: %.3f  lengthscale: %.3f   noise: %.3f' % (
                        i + 1, training_iter, loss.item(),
                        model_gpC.covar_module.base_kernel.lengthscale.item(),
                        model_gpC.likelihood.noise.item()))            
                optimizer.step()        

            # Get into evaluation (predictive posterior) mode
            model_gpC.eval()
            likelihood_gpC.eval()

            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_mean = model_gpC(X_test)
                observed_pred = likelihood_gpC(model_gpC(X_test))
            num_points = Y_test.size()[1]
            mse_gpC = 1.0/num_points*torch.sum(torch.square(Y_test - observed_mean.loc))



            #--------------------------- GP-Linear ---------------------------#
            # initialize likelihood and model
            print('GP-Linear Model') 
            training_iter = kwargs['epochs_GPL']
            likelihood_gpL = gpytorch.likelihoods.GaussianLikelihood()
            model_gpL = LinearGPModel(X_train, Y_train, likelihood_gpL, kwargs['kernel'])

            # Find optimal model hyperparameters - Check if we need this step
            model_gpL.train()
            likelihood_gpL.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(model_gpL.parameters(), lr=kwargs['learning_rate_gpL'])  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_gpL, model_gpL)
            mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood_gpL, model_gpL)

            for i in range(training_iter):
                optimizer.zero_grad()         # Zero gradients from previous iteration
                output = model_gpL(X_train)   # Output from model       
                loss = -mll(output, Y_train)  # Calc loss and backprop gradients 
                loss.backward()
                if self.deep_verbose:
                    print('Iter %d/%d - Loss: %.3f  lengthscale: %.3f   noise: %.3f' % (
                        i + 1, training_iter, loss.item(),
                        model_gpC.covar_module.base_kernel.lengthscale.item(),
                        model_gpC.likelihood.noise.item()))      
                optimizer.step()

            # Get into evaluation (predictive posterior) mode
            model_gpL.eval()
            likelihood_gpL.eval()

            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_mean = model_gpL(X_test)
                observed_pred = likelihood_gpL(model(X_test))
            num_points = Y_test.size()[1]
            mse_gplinear = 1.0/num_points*torch.sum(torch.square(Y_test - observed_mean.loc))


            #--------------------------- GP-NN ---------------------------#
            # initialize likelihood and model
            print('GP-NN Model')
            training_iter = kwargs['epochs_GPNN']
            likelihood_gpnn = gpytorch.likelihoods.GaussianLikelihood()
            model_gpnn = NNGPModel(X_train, Y_train, likelihood_gpnn,
                                     saveModel_filename,kwargs['num_nodes'], kwargs['kernel'])

            # Find optimal model hyperparameters
            model_gpnn.train()
            likelihood_gpnn.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(model_gpnn.parameters(), lr=kwargs['learning_rate_gpNN'])  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_gpnn, model_gpnn)
            mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood_gpnn, model_gpnn)

            for i in range(training_iter):
                optimizer.zero_grad()         # Zero gradients from previous iteration
                output = model_gpnn(X_train)  # Output from model
                loss = -mll(output, Y_train)  # Calc loss and backprop gradients 
                loss.backward()

                if self.deep_verbose:
                    print('Iter %d/%d - Loss: %.3f  lengthscale: %.3f   noise: %.3f' % (
                        i + 1, training_iter, loss.item(),
                        model_gpnn.covar_module.base_kernel.lengthscale.item(),
                        model_gpnn.likelihood.noise.item()))
                optimizer.step()

            # Get into evaluation (predictive posterior) mode
            model_gpnn.eval()
            likelihood_gpnn.eval()

            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_mean = model_gpnn(X_test)
                observed_pred = likelihood_gpnn(model(X_test))
            num_points = Y_test.size()[1]
            mse_gpnn = 1.0/num_points*torch.sum(torch.square(Y_test - observed_mean.loc))

            return mse_gp0, mse_gpcon, mse_gplinear, mse_gpnn, model_gp0, model_gpC, model_gpL, model_gpnn, likelihood_gp0, likelihood_gpC, likelihood_gpL, likelihood_gpnn

        return
    


#A Unified class for all GP Models
class UnifiedGPModel(gpytorch.models.ExactGP,GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API
    MIN_INFERRED_NOISE_LEVEL = 1e-5
    def __init__(self, train_x, train_y, likelihood,ker):
        super().__init__(train_x, train_y, likelihood)
        if ker=='RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif ker=='Matern':            
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

#Sub-class for implementing 0 Mean GP
class ZeroGPModel(UnifiedGPModel):
    def __init__(self, train_x, train_y, likelihood,ker):
        super().__init__(train_x, train_y, likelihood,ker)
        self.mean_module = gpytorch.means.ZeroMean()


#Sub-class for implementing Constant Mean GP
class ConstantGPModel(UnifiedGPModel):
    def __init__(self, train_x, train_y, likelihood,ker):
        super().__init__(train_x, train_y, likelihood,ker)
        self.mean_module = gpytorch.means.ConstantMean()


#Sub-class for implementing Linear Mean GP
class LinearGPModel(UnifiedGPModel):
    def __init__(self, train_x, train_y, likelihood,ker):
        super().__init__(train_x, train_y, likelihood,ker)
        self.mean_module = gpytorch.means.LinearMean(train_x.shape[1], bias=True)


#Sub-class for implementing NN GP
class NNGPModel(UnifiedGPModel):
    def __init__(self, train_x, train_y, likelihood,ker,saveModel_filename, num_nodes):
        super().__init__(train_x, train_y, likelihood,ker)
        self.mean_module = NeuralNetwork(train_x.shape[1],num_nodes)
        self.mean_module.load_state_dict(torch.load(saveModel_filename))
        self.mean_module.eval()
        
    def forward_stand_alone_fit_call(self,x):
        ''' 
        This forward call works when x is a 2-D tensor, 
        Like what is encountered in the initial fitting in BO
        or stand alone fitting
        ''' 
        output_nn = self.mean_module(x) 
        mean_x = torch.flatten(output_nn) 
        covar_x = self.covar_module(x)        
        
        return mean_x, covar_x

    def forward_acq_func_call(self,x):
        ''' 
        This forward call works when x is a 3-D tensor, 
        Like what is encountered when the acquisition function
        in BO calls the GP model
        ''' 
        output_nn = self.mean_module(x) 
        mean_x = output_nn 
        covar_x = self.covar_module(x)           
        
        return mean_x, covar_x
    
    def forward(self, x):

        ''' 
        This forward call works for the GP model, used to make predictions
        ''' 
        mean_x, covar_x = self.forward_stand_alone_fit_call(x)  
        if len(x.shape) == 2:
            mean_x, covar_x = self.forward_stand_alone_fit_call(x)  
        elif len(x.shape) == 3:
            mean_x, covar_x = self.forward_acq_func_call(x)
            mean_x = torch.reshape(mean_x,(mean_x.shape[0],mean_x.shape[1])) # Mean cannot be a 3-D tensor
            
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

#Custom loss class to have the functionality of all losses
class Loss(nn.Module):
    def __init__(self,x,y):
        super(Loss,self).__init__()
        self.pred=x
        self.obs=y
    def MSE(self):
        loss = nn.MSELoss
        return loss(self.pred,self.obs)
    def MAE(self):
        loss = nn.L1Loss
        return loss(self.pred,self.obs)
    def GNLLL(self):
        loss = nn.GaussianNLLLoss
        return loss(self.pred,self.obs)
    def RMSE(self):
        loss = nn.MSELoss
        return torch.sqrt(loss(self.pred,self.obs))
    def MAPE(self):
        return torch.mean(torch.abs((self.obs - self.pred) / self.obs))
