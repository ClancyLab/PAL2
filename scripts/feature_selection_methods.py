import numpy as np     
import csv 
import copy 
import random 
import pandas as pd
import scipy
import sklearn as sk

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error

# User defined files and classes
import utils_dataset as utilsd
import code_verification as verification

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso


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
    data = pd.read_csv('/Users/maitreyeesharma/WORKSPACE/PostDoc/EngChem/HEMI/chan_code/perovs_dft_ml/HSE_data.csv')
    #descriptors indicating composition (formula)
    Comp_desc = pd.DataFrame(data, columns=['K', 'Rb', 'Cs', 'MA', 'FA', 'Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb', 'Cl', 'Br', 'I'])
    #descriptors using elemental properties (ionic radii, density etc.)
    Elem_desc = pd.DataFrame(data, columns=['A_ion_rad', 'A_BP', 'A_MP', 'A_dens', 'A_at_wt', 'A_EA', 'A_IE', 'A_hof', 'A_hov', 'A_En', 
                                            'A_at_num', 'A_period', 'B_ion_rad', 'B_BP', 'B_MP', 'B_dens', 'B_at_wt', 'B_EA', 'B_IE', 'B_hof', 
                                            'B_hov', 'B_En', 'B_at_num', 'B_period', 'X_ion_rad', 'X_BP', 'X_MP', 'X_dens', 'X_at_wt', 
                                            'X_EA', 'X_IE', 'X_hof', 'X_hov', 'X_En', 'X_at_num', 'X_period'])
    #combined descriptors
    All_desc = pd.DataFrame(data, columns=['K', 'Rb', 'Cs', 'MA', 'FA', 'Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb', 'Cl', 'Br', 'I', 'A_ion_rad', 
                                           'A_BP', 'A_MP', 'A_dens', 'A_at_wt', 'A_EA', 'A_IE', 'A_hof', 'A_hov', 'A_En', 'A_at_num', 'A_period', 
                                           'B_ion_rad', 'B_BP', 'B_MP', 'B_dens', 'B_at_wt', 'B_EA', 'B_IE', 'B_hof', 'B_hov', 'B_En', 'B_at_num', 
                                           'B_period', 'X_ion_rad', 'X_BP', 'X_MP', 'X_dens', 'X_at_wt', 'X_EA', 'X_IE', 'X_hof', 'X_hov', 'X_En', 
                                           'X_at_num', 'X_period'])

    HSE_gap_copy = copy.deepcopy(data.Gap.to_numpy())
    YY=HSE_gap_copy.reshape(-1,1)
    XX = copy.deepcopy(Elem_desc.to_numpy())
    
    X_stand = utilsd.standardize_data(XX)
    Y_stand = utilsd.standardize_data(YY)
    
    tests = verification.code_verification
    tests.test_lasso(X_stand,Y_stand,Elem_desc.columns)
    # tests.test_svm(X_stand,Y_stand)
    tests.test_pearson_corr_coeff(X_stand,Y_stand)