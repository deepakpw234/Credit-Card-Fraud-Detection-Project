import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score



@dataclass
class Stage2DataTransformationConfig:
    scaled_data = os.path.join(os.getcwd(),"artifacts")

class Stage2DataTransformation:
    def __init__(self):
        self.stage2_data_transformation_config = Stage2DataTransformationConfig()


    def undersample_splitting(self,scaled_df):
        try:
            scaled_df = scaled_df
            logging.info("Splitting of scaled data into input and target columns is started")

            # Shufflinng the scaled dataset
            scaled_df = scaled_df.sample(frac=1)


            fraud_df = scaled_df[scaled_df["Class"]==1]
            non_fraud_df = scaled_df[scaled_df["Class"]==0].iloc[0:492]

            undersample_df = pd.concat((fraud_df,non_fraud_df),axis=0)
            logging.info("Undersample scaled data is splitted")

            undersample_df = undersample_df.sample(frac=1)

            logging.info("undersample data is reshuffled")

        except Exception as e:
            raise CustomException
        
        return undersample_df
    
    def removing_outlier(self,undersample_df):
        try:
            undersample_df = undersample_df
            logging.info("Removing of outlier from undersample data is started")
            
            correlation = undersample_df.corr()    # This correlation matrix will give the features that have highly influence or impact on our undersample dataset

            # Features V14, V12, V10 are having the high impact on our dataset so we will remove outliers from this features first

            # Outlier removal from V14 feature
            logging.info("Removal from V14")
            v14_df = undersample_df['V14'].loc[undersample_df['Class']==1].values
            q25, q75 = np.percentile(v14_df,25), np.percentile(v14_df,75)
            v14_iqr = q75-q25  # Finding the inter quartile range 

            v14_cutoff = 1.5*v14_iqr

            v14_lower = q25-v14_cutoff
            v14_upper = q75+v14_cutoff

            v14_outlier = []
            for value in v14_df:
                if value < v14_lower or value > v14_upper:
                    v14_outlier.append(value)
        

            undersample_df = undersample_df.drop((undersample_df[((undersample_df['V14']<v14_lower) | (undersample_df['V14']>v14_upper)) & (undersample_df['Class']==1)]).index,axis=0)


            # Outlier removal from V12 feature
            logging.info("Removal from V12")
            v12_df = undersample_df['V12'].loc[undersample_df['Class']==1].values
            q25, q75 = np.percentile(v12_df,25), np.percentile(v12_df,75)
            v12_iqr = q75-q25

            v12_cutoff = 1.5*v12_iqr

            v12_min = q25-v12_cutoff
            v12_max=q75+v12_cutoff

            v12_outlier = []
            for value in v12_df:
                if value < v12_min or value > v12_max:
                    v12_outlier.append(value)

            undersample_df = undersample_df.drop((undersample_df[((undersample_df['V12']<v12_min) | (undersample_df['V12']>v12_max)) & (undersample_df['Class']==1)]).index,axis=0)          


            # Outlier removal from V10 feature
            logging.info("Removal from V10")
            v10_df = undersample_df['V10'].loc[undersample_df['Class']==1].values
            q25, q75 = np.percentile(v10_df,25), np.percentile(v10_df,75)
            v10_iqr = q75-q25

            v10_cutoff = 1.5*v10_iqr

            v10_min = q25-v10_cutoff
            v10_max=q75+v10_cutoff

            v10_outlier = []
            for value in v10_df:
                if value < v10_min or value > v10_max:
                    v10_outlier.append(value)
            
            undersample_df = undersample_df.drop((undersample_df[((undersample_df['V10']<v10_min) | (undersample_df['V10']>v10_max)) & (undersample_df['Class']==1)]).index,axis=0)
            
            logging.info("Outlier removal from undersample data is completed")

        except Exception as e:
            raise CustomException(e,sys)
        
        return undersample_df
    

        


