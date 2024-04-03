import numpy as np     
import csv 
import copy 
import random 
import pandas as pd
import scipy
import os

import xgboost as xgb
from xgboost.sklearn import XGBRegressor

import sklearn as sk
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso

# User defined files and classes
import input_class
import utils_dataset as utilsd
import code_verification as verification

# Plotting
import matplotlib.pyplot as plt

# Data reading specific modules
import json
import ruamel.yaml

# Write log files
import sys

class feature_selection_algorithms:

    def __init__(self,XX,YY,test_size=0.33,random_state=42):
        
        # Train Data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(XX, YY, test_size=test_size, random_state=random_state)
        
    # LASSO 
    def lasso(self,alpha_range=np.arange(0.001,0.1,0.001)):
        pipeline = Pipeline([('scaler',StandardScaler()),('model',Lasso())])
        search = GridSearchCV(pipeline,{'model__alpha':alpha_range},cv = 5, scoring="neg_mean_squared_error")
        search.fit(self.X_train,self.y_train)
        lasso_parameters = search.best_params_
        coefficients = search.best_estimator_.named_steps['model'].coef_
        
        return search, lasso_parameters, coefficients
    
    # XGBoost
    def xgboost(self, **kwargs):
        
        clf = XGBRegressor(n_estimators=200, learning_rate=0.025, max_depth=20, verbosity=0, booster='gbtree', 
                    reg_alpha=np.exp(-6.788644799030888), reg_lambda=np.exp(-7.450413274554533), 
                    gamma=np.exp(-5.374463422208394), subsample=0.5, objective= 'reg:squarederror', n_jobs=1)  
        
        paras = clf.get_params()

        clf.fit(self.X_train, self.y_train)        
        return clf
    
    # SVM
    def svm_unsupervised(self):
        
        clf = sk.svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(self.X_train)
        y_pred_train = clf.predict(self.X_train)
        n_error_train = y_pred_train[y_pred_train == -1].size

        return self.X_train, y_pred_train, n_error_train
    
    # SVM
    def svm_supervised(self,y_labels,kernel="poly",degree=3,gamma=0.1):
        
        clf = sk.svm.OneClassSVM(nu=0.1, kernel=kernel, gamma=gamma)
        clf.fit(self.X_train,y_labels)

        return clf
    
    # Pearson correlation coefficients
    def pearson_corr_coeff(self,x,y):
        
        XX_transpose = np.transpose(x)
        rho_coeff=[]
        for i in range(0,np.size(x,axis=1)):
            rho_coeff_temp = scipy.stats.pearsonr(XX_transpose[i], y)
            rho_coeff.append(rho_coeff_temp.pvalue)
            
        return rho_coeff
    
    # Features selected by XGBoost
    def selected_features_xgboost(self, descriptors, deep_verbose=False):
        
        clf = self.xgboost()
        score = clf.score(self.X_train, self.y_train)
        if deep_verbose:
            print("XGBoost Training score: ", score)

        scores = cross_val_score(clf, self.X_train, self.y_train,cv=10)
        if deep_verbose:
            print("XGBoost Mean cross-validation score: %.2f" % scores.mean())


        ypred = clf.predict(self.X_test)
        mse = mean_squared_error(self.y_test, ypred)
        if deep_verbose:
            print("XGBoost MSE: %.2f" % mse)
            print("XGBoost RMSE: %.2f" % (mse**(1/2.0)))

        f_importance = clf.get_booster().get_score(importance_type='gain')
        feature_importance_dict={}

        for f,value in f_importance.items():
            feature_index = int(f.split('f')[1])
            feature_importance_dict[descriptors[feature_index]] = value
            if deep_verbose:
                print(f"Column: {feature_index}, descriptor: {descriptors[feature_index]}")
            
        return feature_importance_dict.keys()
                
    
if __name__=="__main__":
    sys.stdout = open(snakemake.log[0],'w')
    
    #Reading user input values
    with open(snakemake.input[0],'r') as f:
        input_dict=ruamel.yaml.safe_load(f)

    input_type = input_dict['InputType']
    input_path = input_dict['InputPath'][0]+input_dict['InputPath'][1]
    input_file = input_dict['InputFile']
    add_target_noise = input_dict['AddTargetNoise']
    test_size = input_dict['test_size_fs']
    random_state = input_dict['random_state']
    onlyImportant = input_dict['onlyImportant']
    output_dir = input_dict['output_folder'][0]+input_dict['output_folder'][1]

    input = input_class.inputs(input_type=input_type,
                                input_path=input_path,
                                input_file=input_file,
                                add_target_noise=add_target_noise)

    XX, YY, descriptors = input.read_inputs(input_dict['verbose'])

    # Transforming datasets by standardization
    if input_dict['standardize_data']:
        X_stand, scalerX_transform = utilsd.standardize_data(XX)
        Y_stand, scalerY_transform = utilsd.standardize_data(YY)
    else:
        X_stand=XX.to_numpy()
        Y_stand = YY

    tests = verification.code_verification


    ## Create new output dir to write all output:
    newpath = output_dir + 'feature_engineering/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    ## Adjust plots format
    plt.tight_layout()

    ## Lasso:
    lasso_df, model = tests.test_lasso(X_stand,Y_stand,descriptors, onlyImportant = onlyImportant, test_size = test_size, random_state = random_state)
    # lasso_df.to_csv(newpath + snakemake.output[0], index=True)
    lasso_df.to_csv(newpath + 'lasso_data.csv', index=True)
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111)
    ax.bar(descriptors,lasso_df[0])
    ax.set_ylabel('Descriptor Importance')
    plt.xticks(range(len(descriptors)),descriptors,rotation=90)
    plt.rcParams.update({'font.size': 20})
    ax.tick_params(direction='out')
    # Adjust layout
    plt.tight_layout()
    plt.savefig(newpath + 'lasso_plot.pdf', bbox_inches='tight')

    plt.show()
    
    # with open(newpath + snakemake.output[2], 'w') as f:
    #     json.dump(model, f)
    with open(newpath + 'lasso_model.json', 'w') as f:
        json.dump(model, f)


    ## XGboost:
    xgboost_df, clf = tests.test_xgboost(X_stand,Y_stand,descriptors, onlyImportant = onlyImportant, test_size = test_size, random_state = random_state)
    # xgboost_df.to_csv(newpath + snakemake.output[3],index=True)
    xgboost_df.to_csv(newpath + 'xgboost_data.csv',index=True)
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111)
    ax.bar(descriptors,xgboost_df[0])
    ax.set_ylabel('Descriptor Importance')
    plt.xticks(range(len(descriptors)),descriptors,rotation=90)
    plt.rcParams.update({'font.size': 20})
    ax.tick_params(direction='out')
    # Adjust layout
    plt.tight_layout()
    plt.savefig(newpath + 'xgboost_plot.pdf', bbox_inches='tight')

    plt.show()
    
    # clf.save_model(newpath + snakemake.output[5])
    clf.save_model(newpath + 'xgboost_model.json')


    ## Pearson:
    pearson_df = tests.test_pearson_corr_coeff(X_stand,Y_stand,descriptors, onlyImportant = onlyImportant, test_size = test_size, random_state = random_state)
    # pearson_df.to_csv(newpath + snakemake.output[6],index=True)
    pearson_df.to_csv(newpath + 'pearson_data.csv',index=True)
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111)
    ax.bar(descriptors,pearson_df[0])
    ax.set_ylabel('Descriptor Importance')
    plt.xticks(range(len(descriptors)),descriptors,rotation=90)
    plt.rcParams.update({'font.size': 20})
    ax.tick_params(direction='out')
    # Adjust layout
    plt.tight_layout()
    plt.savefig(newpath + 'pearson_plot.pdf', bbox_inches='tight')

    plt.show()
