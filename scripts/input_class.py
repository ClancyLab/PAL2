import sklearn
import numpy as np     
import csv 
import copy 
import random 
import pandas as pd
import pickle
import json  
import openpyxl
import itertools

# User defined files and classes
import feature_selection_methods as feature_selection
import utils_dataset as utilsd
import code_verification as verification

# Plotting libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

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

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 15


class inputs:
    def __init__(self,input_type='PerovAlloys',input_path='/Users/maitreyeesharma/WORKSPACE/PostDoc/EngChem/HEMI/chan_code/perovs_dft_ml/',input_file='HSE_data.csv',add_target_noise = False):
        self.input_type = input_type
        self.input_path = input_path
        self.input_file = input_file
        self.add_target_noise = add_target_noise
        self.filename   = self.input_path + self.input_file
        
    def read_inputs(self, verbose):
        if verbose:
            print('Reading data for the input dataset type: ', self.input_type)
        
        # Add options for different datasets that we want to read
        if self.input_type == 'PerovAlloys':
            XX, YY, descriptors = self.read_perovalloys()
        elif self.input_type == 'PALSearch':
            XX, YY, descriptors = self.read_palsearch()
        elif self.input_type == 'Gryffin':
            XX, YY, descriptors = self.read_gryffin()            
        elif self.input_type == 'MPEA':
            XX, YY, descriptors = self.read_MPEA()
        elif self.input_type == 'connor-polymers':
            XX, YY, descriptors = self.read_connor_polymers()
        return XX, YY, descriptors
    
#    def read_ClancyLab(self):

#        XX ---> Inputs to your surrogate
#        YY --> Target material property
#        descriptors --> String with physiochemical properties of XX.
        
#        return XX, YY, descriptors
    
    def read_MPEA(self):
        '''
        This function reads the dataset from the HEA review paper: https://www.nature.com/articles/s41597-020-00768-9
        input_type='MPEA',
        input_path='/Users/maitreyeesharma/WORKSPACE/PostDoc/EngChem/Space@Hopkins_HEA/dataset/',
        input_file='curated_MPEA_initial_hardness_value.csv'
        '''     
        data = pd.read_csv(self.filename)
 
        input_composition_cols = data.columns[0:29]
        input_property_cols = data.columns[30:35]
        input_composition_df = pd.DataFrame(data, columns=['Ti', 'Pd', 'Ga', 'Al', 'Co', 'Si', 'Mo', 'Sc', 'Zn', 'C', 'Sn', 'Nb', 'Ag', 'Mg', 'Mn', 'Y', 
                                    'Re', 'W', 'Zr', 'Ta', 'Fe', 'Cr', 'B', 'Cu', 'Hf', 'Li', 'V', 'Nd', 'Ni', 'Ca'])

        XX = pd.DataFrame(data, columns=input_property_cols)
        descriptors = input_property_cols
        target = copy.deepcopy(data['Target'].to_numpy())
        YY = target.reshape(-1,1)

        return XX, YY, descriptors
        
    def read_perovalloys(self):
        '''
        This function reads the dataset from Maria Chan's paper: https://doi.org/10.1039/D1EE02971A
        It does not need any inputs. 
        Inputs to the class for this case are the default class inputs
        '''
        
        data = pd.read_csv(self.filename)

        Comp_desc = pd.DataFrame(data, columns=['K', 'Rb', 'Cs', 'MA', 'FA', 'Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb', 'Cl', 'Br', 'I'])

        # Descriptors using elemental properties (ionic radii, density etc.) shortlisted
        perov_desc_shortlist = pd.DataFrame(data, columns=['A_ion_rad', 'A_dens', 'A_at_wt', 'A_EA', 'A_IE', 'A_En', 
                                                'B_ion_rad', 'B_dens', 'B_at_wt', 'B_EA', 'B_IE', 'B_En', 
                                                'X_ion_rad', 'X_dens', 'X_at_wt', 'X_EA', 'X_IE', 'X_En', 'X_at_num'])

        # Descriptors using elemental properties (ionic radii, density etc.) shortlisted
        perov_desc_forNikhil = pd.DataFrame(data, columns=['A_ion_rad', 'A_EA', 'A_IE', 'A_En', 
                                                'B_ion_rad', 'B_EA', 'B_IE', 'B_En', 
                                                'X_ion_rad', 'X_EA', 'X_IE', 'X_En'])
        
        # Descriptors using elemental properties (ionic radii, density etc.) same as the paper
        perov_desc_paper = pd.DataFrame(data, columns=['A_ion_rad', 'A_BP', 'A_MP', 'A_dens', 'A_at_wt', 'A_EA', 'A_IE', 'A_hof', 'A_hov', 'A_En', 
                                            'A_at_num', 'A_period', 'B_ion_rad', 'B_BP', 'B_MP', 'B_dens', 'B_at_wt', 'B_EA', 'B_IE', 'B_hof', 
                                            'B_hov', 'B_En', 'B_at_num', 'B_period', 'X_ion_rad', 'X_BP', 'X_MP', 'X_dens', 'X_at_wt', 
                                            'X_EA', 'X_IE', 'X_hof', 'X_hov', 'X_En', 'X_at_num', 'X_period'])
        
        # All descriptors
        All_desc = pd.DataFrame(data, columns=['K', 'Rb', 'Cs', 'MA', 'FA', 'Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb', 'Cl', 'Br', 'I', 'A_ion_rad', 
                                           'A_BP', 'A_MP', 'A_dens', 'A_at_wt', 'A_EA', 'A_IE', 'A_hof', 'A_hov', 'A_En', 'A_at_num', 'A_period', 
                                           'B_ion_rad', 'B_BP', 'B_MP', 'B_dens', 'B_at_wt', 'B_EA', 'B_IE', 'B_hof', 'B_hov', 'B_En', 'B_at_num', 
                                           'B_period', 'X_ion_rad', 'X_BP', 'X_MP', 'X_dens', 'X_at_wt', 'X_EA', 'X_IE', 'X_hof', 'X_hov', 'X_En', 
                                           'X_at_num', 'X_period'])
    
        descriptors = perov_desc_shortlist.columns
        XX = pd.DataFrame(data, columns=descriptors)
#         XX = copy.deepcopy(perov_desc_forNikhil.to_numpy())
#         descriptors = perov_desc_forNikhil.columns
    
        HSE_gap_copy = copy.deepcopy(data.Gap.to_numpy())
        YY = HSE_gap_copy.reshape(-1,1)
        
        return XX, YY, descriptors

    def read_palsearch(self):
        '''
        This function reads the datasets built for PAL-Search
        It does not need any inputs. 
        input_type='PALSearch',
        input_path='/Users/maitreyeesharma/WORKSPACE/PostDoc/EngChem/PAL_Search/Datasets/',
        input_file='test_r2.xlsx'
        '''
        xls = pd.ExcelFile(self.filename)
        Data_DF1 = pd.read_excel(xls, 'ALL_RESULTS_DATA')
        Data_DF2 = pd.read_excel(xls, 'PROPERTY_BASKET')
        initial_cols = len(Data_DF1.columns)-1
        descriptors = []
        for num_col_df1 in range(0,initial_cols):
            bridge_dict = {}
            bridge = [element for element in Data_DF2.columns if Data_DF1.columns[num_col_df1] in element and 'CHOICES' not in element]
            descriptors.append(bridge)
            choices = [element for element in Data_DF2[Data_DF1.columns[num_col_df1]+'-CHOICES'] if str(element) != "nan"]
            list_features = []

            if bool(bridge):
                for column in bridge:
                    list_features_temp = Data_DF2[column][0:len(choices)].to_numpy().tolist()
                    list_features.append(list_features_temp)

                list_np = np.transpose(np.array(list_features))
                i=0
                for choice in choices:
                    bridge_dict[choice] = list_np[i]
                    i = i+1

                bridge_property_add = []
                for bridge_choice in Data_DF1[Data_DF1.columns[num_col_df1]]:

                    bridge_property_add_temp = bridge_dict[bridge_choice]
                    bridge_property_add.append(bridge_property_add_temp)   

                bridge_property_add = np.transpose(np.reshape(bridge_property_add,[len(Data_DF1[Data_DF1.columns[0]]),len(bridge)]))

                for column_num in range(0,len(bridge)):
                    Data_DF1[bridge[column_num]] = bridge_property_add[column_num]
        
        descriptors = np.array(list(itertools.chain.from_iterable(descriptors)),dtype='object')
        XX = pd.DataFrame(Data_DF1, columns=descriptors)

        target = copy.deepcopy(Data_DF1.Target.to_numpy())
        YY = target.reshape(-1,1)
        
        ## Adding noise to reactions 1-3 data
        if self.add_target_noise == 'True':
            if not "ALL_RESULTS_DATA_withNoise" in xls.sheet_names:
                YY_noise = np.random.normal(0,0.05,np.size(YY)).reshape(np.size(YY),1)
                YY = YY + YY_noise
                Data_DF1 = Data_DF1.drop(columns=['Target'])
                Data_DF1['Target'] = YY                
                writer = pd.ExcelWriter(self.filename, 'openpyxl', mode='a')
                Data_DF1.to_excel(writer, sheet_name='ALL_RESULTS_DATA_withNoise',index=False)
                writer.close()  
            elif "ALL_RESULTS_DATA_withNoise" in xls.sheet_names:
                Data_DF_target = pd.read_excel(xls, 'ALL_RESULTS_DATA_withNoise')
                target = copy.deepcopy(Data_DF_target.Target.to_numpy())
                YY = target.reshape(-1,1)
        
        print(XX,YY,descriptors)
        return XX, YY, descriptors
    
    def read_gryffin(self):
        '''
        This function reads the perovskite dataset used in the GRYFFIN paper: https://doi.org/10.1063/5.0048164
        It does not need any inputs. 
        Inputs to the class for this case should be: 
        input_type='Gryffin',
        input_path='/Users/maitreyeesharma/WORKSPACE/PostDoc/EngChem/MatDisc_ML/datasets/',
        input_file='perovskites_GRYFFIN.pkl'
        '''
        
        lookup_df = pickle.load(open(self.filename, 'rb'))
        perov_desc = pd.DataFrame(lookup_df, columns=['organic-homo',
                                   'organic-lumo', 'organic-dipole', 'organic-atomization',
                                   'organic-r_gyr', 'organic-total_mass', 'anion-electron_affinity',
                                   'anion-ionization_energy', 'anion-mass', 'anion-electronegativity',
                                   'cation-electron_affinity', 'cation-ionization_energy', 'cation-mass',
                                   'cation-electronegativity'])
        
        XX = perov_desc
        HSE_gap_copy = copy.deepcopy(lookup_df.hse06.to_numpy())
        YY=HSE_gap_copy.reshape(-1,1)
        # YY = -1.0*YY
        
        return XX, YY, perov_desc.columns


    def read_connor_polymers(self):
        xls = pd.ExcelFile(self.filename)
        df = pd.read_excel(xls, 'pal-dimer+1solv-388')

        XX = df.drop(columns=['Solvent-Label', 'Polymer-Label', 'Identifier', 'Target (eV)'], axis=1)
        YY = df['Target (eV)'].to_numpy().reshape(-1,1)

        descriptors = np.array(df.columns[4:])

        return XX, YY, descriptors
    
if __name__=="__main__":
    
    run_folder = '/Users/maitreyeesharma/WORKSPACE/PostDoc/EngChem/MatDisc_ML/feature_engineering/'
    with open(run_folder+'inputs.json', "r") as f:
        input_dict = json.load(f)
    
    input_type = input_dict['InputType']
    input_path = input_dict['InputPath']
    input_file = input_dict['InputFile']
    
    # input = inputs(input_type=input_type,input_path=input_path,input_file=input_file)
    input = inputs(input_type='PALSearch',input_path='/Users/maitreyeesharma/WORKSPACE/PostDoc/EngChem/PAL_Search/Datasets/',input_file='reaction_1.xlsx')
    # input=inputs()
    XX, YY, descriptors = input.read_inputs()
    
    X_stand = utilsd.standardize_data(XX)
    Y_stand = utilsd.standardize_data(YY)
    
    tests = verification.code_verification
    tests.test_lasso(X_stand,Y_stand,descriptors,onlyImportant=False)
    # tests.test_svm(Y_stand,Y_stand)
    # tests.test_pearson_corr_coeff(X_stand,Y_stand,descriptors,onlyImportant=False)
    tests.test_xgboost(X_stand,Y_stand,descriptors,onlyImportant=False)
    
