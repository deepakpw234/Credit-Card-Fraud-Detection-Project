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
    


class Stage1DataTransformation:
    def __init__(self):
        self.stage1_data_transfarmation_config = Stage1DataTransformationConfig()

    
    def get_scaled_data(self,unscaled_dataset_file_path):
        try:
            '''
            By observing the dataset, it is found that time and amount columns 
            are not scaled up so we will first scaled up these columns by RobustScaler. 
            RobustScaler is used here as it good for with outlier '''

            unscaled_df = pd.read_csv(unscaled_dataset_file_path)
            print(unscaled_df)



        except Exception as e:
            raise CustomException(e,sys)
        
