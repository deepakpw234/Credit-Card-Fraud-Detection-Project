import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import RobustScaler


@dataclass
class Stage1DataTransformationConfig:
    scaled_dataset = os.path.join(os.getcwd(),"artifacts")
    unscaled_dataset = os.path.join(os.getcwd(),"artifacts","creditcard.csv")


class Stage1DataTransformation:
    def __init__(self):
        self.stage1_data_transfarmation_config = Stage1DataTransformationConfig()

    
    def get_scaled_data(self,unscaled_dataset_path):
        try:
            '''
            By observing the dataset, it is found that time and amount columns 
            are not scaled up so we will first scaled up these columns by RobustScaler. 
            RobustScaler is used here as it good for with outlier '''

            logging.info("Dataset Scaling is started")

            unscaled_df = pd.read_csv(unscaled_dataset_path)
            # print(unscaled_df)
            logging.info("Unscaled dataset is loaded from creditcard.csv")

            robust = RobustScaler()
            Amout_df = unscaled_df[["Amount"]]
            Time_df = unscaled_df[["Time"]]
    

            Amout_df_scaled = robust.fit_transform(Amout_df)
            Time_df_scaled = robust.fit_transform(Time_df)

            logging.info("Amount and Time are scaled up")

            Amout_df_scaled = pd.DataFrame(Amout_df_scaled,columns=["scaled_amount"])
            Time_df_scaled = pd.DataFrame(Time_df_scaled,columns=["scaled_time"])
            
            # Dropping the unscaled Time and Amount columns from original datset
            unscaled_df = unscaled_df.drop(["Amount","Time"],axis=1)
            logging.info("unsacled amount and time is drop")


            scaled_df = pd.concat((Amout_df_scaled,Time_df_scaled,unscaled_df),axis=1)

            logging.info("Scaled dataframe is created")


        except Exception as e:
            raise CustomException(e,sys)
        
        return scaled_df
        

