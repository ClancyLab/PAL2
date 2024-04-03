import sklearn
import numpy as np     
import csv 
import copy 
import random 
import pandas as pd
import pickle

# Scikit learn packages for model fitting and scores
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# User defined files and classes
import feature_selection_methods as feature_selection

import matplotlib as mpl
import matplotlib.pyplot as plt

class code_verification:
    def __init__():
        
        print('This class is build to test various implementations in the code')
        
    
    # Testing if the input format and computation of LASSO is correct
    def test_lasso(X_stand, Y_stand, descriptors, onlyImportant=True, test_size = 0.2, random_state = 40):
        
        
        fs = feature_selection.feature_selection_algorithms(X_stand,Y_stand,test_size=test_size,random_state=random_state)
        
        search, lasso_parameters, coefficients = fs.lasso(alpha_range=np.arange(0.0001,0.01,0.0001))
        other_data = search.best_params_
        coefficients = search.best_estimator_.named_steps['model'].coef_
        importance = np.abs(coefficients)
        feature_importance_dict = {}

        
        if onlyImportant:
            dict_keys = descriptors[importance > 0.0]
            dict_values = importance[importance > 0.0]
            for i in range(0,len(dict_keys)):
                feature_importance_dict[dict_keys[i]] = dict_values[i]
        elif not onlyImportant:
            for i in range(0,len(descriptors)):
                feature_importance_dict[descriptors[i]] = importance[i]
            
        importance_df = pd.DataFrame.from_dict(data=feature_importance_dict, orient='index')
        # other_df = pd.DataFrame.from_dict(data=other_data, orient='index')
        return importance_df, other_data
    
    # Testing if the input format and computation of SVM is correct
    def test_svm(X_stand,Y_stand):
        
        fs = feature_selection.feature_selection_algorithms(X_stand,Y_stand,test_size=0.2,random_state=40)
        
        X_train, y_pred, error = fs.svm_unsupervised()
        fig3, ax3 = plt.subplots(figsize=(8,5.3))
        c = X_train
        ax3.scatter(X_train.transpose(),y_pred,c=c)
        
        return
        
    # Testing if the input format and computation of pearson correlation coefficients is correct
    def test_pearson_corr_coeff(X_stand, Y_stand, descriptors, onlyImportant=True, test_size = 0.2, random_state = 40):
        
        fs = feature_selection.feature_selection_algorithms(X_stand,Y_stand,test_size=test_size,random_state=random_state)
        Y_stand = Y_stand.reshape(len(Y_stand))
        rho = fs.pearson_corr_coeff(X_stand,Y_stand)
        
        feature_importance_dict={}
        
        if onlyImportant:
            for f in range(0,len(descriptors)):
                if rho[f] > 1.e-2:
                    feature_importance_dict[descriptors[f]] = rho[f]  
        if not onlyImportant:
            for f in range(0,len(descriptors)):
                feature_importance_dict[descriptors[f]] = rho[f] 
            
        importance_df = pd.DataFrame.from_dict(data=feature_importance_dict, orient='index')
        
        return importance_df
    
    def test_xgboost(X_stand, Y_stand, descriptors, onlyImportant=True, test_size = 0.2, random_state = 40):
        fs = feature_selection.feature_selection_algorithms(X_stand,Y_stand,test_size=test_size,random_state=random_state)
        
        other_data = dict()
        clf = fs.xgboost()
        score = clf.score(fs.X_train, fs.y_train)
        other_data['training_score'] = score
        # print("Training score: ", score)

        scores = cross_val_score(clf, fs.X_train, fs.y_train,cv=10)
        other_data['cross_val_score'] = scores.mean()
        # print("Mean cross-validation score: %.2f" % scores.mean())


        ypred = clf.predict(fs.X_test)
        mse = mean_squared_error(fs.y_test, ypred)
        other_data['MSE'] = mse
        other_data['RMSE'] = mse**(1/2.0)
        # print("MSE: %.2f" % mse)
        # print("RMSE: %.2f" % (mse**(1/2.0)))

        f_importance = clf.get_booster().get_score(importance_type='gain')
        feature_importance_dict={}
        
        if onlyImportant:
            for f,value in f_importance.items():
                feature_index = int(f.split('f')[1])
                feature_importance_dict[descriptors[feature_index]] = value
                # print(f"Column: {feature_index}, descriptor: {descriptors[feature_index]}")
        
        # XGBoost gives scores only for features that were retained
        # The following peice of code sets the score to 0 for the remaining features
        elif not onlyImportant:  
            num_features = np.linspace(0,len(descriptors)-1,len(descriptors), dtype=int)
            num_features_found = []

            for f,value in f_importance.items():
                feature_index = int(f.split('f')[1])
                num_features_found.append(feature_index)       

            num_features_notFound = np.setdiff1d(num_features,num_features_found).tolist()

            for f in num_features_notFound:
                f_importance['f'+str(f)] = 0.0

            for f in num_features:
                feature_importance_dict[descriptors[f]] = f_importance['f'+str(f)]       
        
        importance_df = pd.DataFrame.from_dict(data=feature_importance_dict, orient='index')
        other_df = pd.DataFrame.from_dict(data=other_data, orient='index')
        # importance_df.plot.bar(logy=False)
        
        return importance_df, clf