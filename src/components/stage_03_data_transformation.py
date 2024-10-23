import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from imblearn.over_sampling import SMOTE

@dataclass
class Stage3DataTransformationConfig:
    oversample_data = os.path.join(os.getcwd(),"artifacts")



class Stage3DataTransformation:
    def __init__(self):
        self.stage3_data_transformation_config = Stage3DataTransformationConfig()

    def oversample_splitting(self,scaled_df):
        try:
            scaled_df = scaled_df

            # Shufflinng the scaled dataset
            scaled_df = scaled_df.sample(frac=1)

            fraud_df = scaled_df[["Class"]]
            non_fraud_df = scaled_df.drop("Class",axis=1)

            sm = SMOTE(sampling_strategy="minority",random_state=42)

            Xsm, ysm = sm.fit_resample(non_fraud_df,fraud_df)

            # print(Xsm.shape)
            # print(ysm.shape)

            oversample_df = pd.concat((Xsm,ysm),axis=1)

            oversample_df = oversample_df.sample(frac=1)

            # print(oversample_df)


        except Exception as e:
            raise CustomException(e,sys)
        
        return oversample_df
    

    def oversample_outlier_removal(self, oversample_df):
        try:
            smote_df = oversample_df

            correlation = smote_df.corr()    # This correlation matrix will give the features that have highly influence or impact on our undersample dataset

            # Features V14, V12, V10 are having the high impact on our dataset so we will remove outliers from this features first

            # Outlier removal from V14 feature
            v14_df = smote_df['V14'].loc[smote_df['Class']==1].values
            q25, q75 = np.percentile(v14_df,25), np.percentile(v14_df,75)
            # print(f"25 quantile : {q25} and 75 quantile: {q75}")
            v14_iqr = q75-q25
            # print(f"Inter quantile range: {v14_iqr}")

            v14_cutoff = 1.5*v14_iqr
            # print(f"The cutoff for outlier v14: {v14_cutoff}")

            v14_min = q25-v14_cutoff
            v14_max=q75+v14_cutoff
            # print(f"minimum cutoff: {v14_min} , maximum cutoff: {v14_max}")

            v14_outlier = []
            for value in v14_df:
                if value < v14_min or value > v14_max:
                    v14_outlier.append(value)
            # print(f"The number of outliers in v14 is: {len(v14_outlier)}")
            # print(f"Outliers are: {v14_outlier}")
            # print(len(v14_df))
            smote_df = smote_df.drop((smote_df[((smote_df['V14']<v14_min) | (smote_df['V14']>v14_max)) & (smote_df['Class']==1)]).index,axis=0)



            # Outlier removal from V12 feature
            v12_df = smote_df['V12'].loc[smote_df['Class']==1].values
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
            # print(len(v12_df))
            smote_df = smote_df.drop((smote_df[((smote_df['V12']<v12_min) | (smote_df['V12']>v12_max)) & (smote_df['Class']==1)]).index,axis=0)




            # Outlier removal from V10 feature
            v10_df = smote_df['V10'].loc[smote_df['Class']==1].values
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
            # print(len(v10_df))
            smote_df = smote_df.drop((smote_df[((smote_df['V10']<v10_min) | (smote_df['V10']>v10_max)) & (smote_df['Class']==1)]).index,axis=0)


            print(smote_df)

        except Exception as e:
            raise CustomException(e,sys)
        
        return smote_df
