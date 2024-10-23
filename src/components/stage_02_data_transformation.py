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
            

            # Shufflinng the scaled dataset
            scaled_df = scaled_df.sample(frac=1)


            fraud_df = scaled_df[scaled_df["Class"]==1]
            non_fraud_df = scaled_df[scaled_df["Class"]==0].iloc[0:492]

            test_x = scaled_df.drop("Class",axis=1).iloc[0:40000]
            test_y = scaled_df["Class"].iloc[0:40000]

            undersample_df = pd.concat((fraud_df,non_fraud_df),axis=0)


            undersample_df = undersample_df.sample(frac=1)

            # print(undersample_df)

        except Exception as e:
            raise CustomException
        
        return undersample_df, test_x, test_y
    
    def removing_outlier(self,undersample_df):
        try:
            undersample_df = undersample_df
            
            correlation = undersample_df.corr()    # This correlation matrix will give the features that have highly influence or impact on our undersample dataset

            # Features V14, V12, V10 are having the high impact on our dataset so we will remove outliers from this features first

            # Outlier removal from V14 feature
            v14_df = undersample_df['V14'].loc[undersample_df['Class']==1].values
            q25, q75 = np.percentile(v14_df,25), np.percentile(v14_df,75)
            # print(f"25 quantile : {q25} and 75 quantile: {q75}")
            v14_iqr = q75-q25  # Finding the inter quartile range 
            # print(f"Inter quantile range: {v14_iqr}")

            v14_cutoff = 1.5*v14_iqr
            # print(f"The cutoff for outlier v14: {v14_cutoff}")

            v14_lower = q25-v14_cutoff
            v14_upper = q75+v14_cutoff
            # print(f"minimum cutoff: {v14_lower} , maximum cutoff: {v14_upper}")

            v14_outlier = []
            for value in v14_df:
                if value < v14_lower or value > v14_upper:
                    v14_outlier.append(value)
            # print(f"The number of outliers in v14 is: {len(v14_outlier)}")
            # print(f"Outliers are: {v14_outlier}")
        

            undersample_df = undersample_df.drop((undersample_df[((undersample_df['V14']<v14_lower) | (undersample_df['V14']>v14_upper)) & (undersample_df['Class']==1)]).index,axis=0)
            # print(undersample_df)


            # Outlier removal from V12 feature
            v12_df = undersample_df['V12'].loc[undersample_df['Class']==1].values
            q25, q75 = np.percentile(v12_df,25), np.percentile(v12_df,75)
            # print(f"25 quantile : {q25} and 75 quantile: {q75}")
            v12_iqr = q75-q25
            # print(f"Inter quantile range: {v12_iqr}")

            v12_cutoff = 1.5*v12_iqr
            # print(f"The cutoff for outlier v12: {v12_cutoff}")

            v12_min = q25-v12_cutoff
            v12_max=q75+v12_cutoff
            # print(f"minimum cutoff: {v12_min} , maximum cutoff: {v12_max}")

            v12_outlier = []
            for value in v12_df:
                if value < v12_min or value > v12_max:
                    v12_outlier.append(value)
            # print(f"The number of outliers in v12 is: {len(v12_outlier)}")
            # print(f"Outliers are: {v12_outlier}")

            undersample_df = undersample_df.drop((undersample_df[((undersample_df['V12']<v12_min) | (undersample_df['V12']>v12_max)) & (undersample_df['Class']==1)]).index,axis=0)
            # print(undersample_df)            


            # Outlier removal from V10 feature
            v10_df = undersample_df['V10'].loc[undersample_df['Class']==1].values
            q25, q75 = np.percentile(v10_df,25), np.percentile(v10_df,75)
            # print(f"25 quantile : {q25} and 75 quantile: {q75}")
            v10_iqr = q75-q25
            # print(f"Inter quantile range: {v10_iqr}")

            v10_cutoff = 1.5*v10_iqr
            # print(f"The cutoff for outlier v10: {v10_cutoff}")

            v10_min = q25-v10_cutoff
            v10_max=q75+v10_cutoff
            # print(f"minimum cutoff: {v10_min} , maximum cutoff: {v10_max}")

            v10_outlier = []
            for value in v10_df:
                if value < v10_min or value > v10_max:
                    v10_outlier.append(value)
            # print(f"The number of outliers in v10 is: {len(v10_outlier)}")
            # print(f"Outliers are: {v10_outlier}")
            
            undersample_df = undersample_df.drop((undersample_df[((undersample_df['V10']<v10_min) | (undersample_df['V10']>v10_max)) & (undersample_df['Class']==1)]).index,axis=0)
            # print(undersample_df)
            # print(len(undersample_df))
            

        except Exception as e:
            raise CustomException(e,sys)
        
        return undersample_df
    

        


