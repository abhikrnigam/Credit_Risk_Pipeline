
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

class Preprocessing:

    def __init__(self):
        pass

## Kaggle Credit Risk Data 
## column Home : ['RENT' 'OWN' 'MORTGAGE' 'OTHER']
## column Intent : ['PERSONAL' 'EDUCATION' 'MEDICAL' 'VENTURE' 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
## column Default : ['Y','N']

    def DataPreprocessing(self):
        
        ## Reading the data
        data = pd.read_csv('data/credit_risk.csv')
        print(f"Data loaded successfully")

        ## Changing the categorical values to Numerical values
        le = LabelEncoder()
        data['Home'] = le.fit_transform(data['Home'])
        data['Intent'] = le.fit_transform(data['Intent'])
        data['Default'] = le.fit_transform(data['Default'])

        ## Managing the null values for the data
        ## for Emp_length and Rate
        ## We will use Simple Imputer with strategy as 'mean'
        imputer = SimpleImputer()
        data_new = pd.DataFrame(imputer.fit_transform(data),columns=data.columns)
        
        ## Dropping the irrelevant columns
        data_new = data_new.drop(columns=['Id'])
        data_new = data_new.rename(columns={"Default":"Target"})

        data_new['Target'] = data_new['Target'].astype(int)
        print(f"Data cleaned successfully")
        
        return data_new
        

        

