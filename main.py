import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging


from src.components.data_ingestion import DataIngestion
from src.components.stage_01_data_transformation import Stage1DataTransformation
from src.components.stage_02_data_transformation import Stage2DataTransformation
from src.components.model_training_01 import Stage1ModelTraining

from src.components.stage_03_data_transformation import Stage3DataTransformation
from src.components.model_training_02 import Stage2ModelTraining



if __name__=="__main__":
    data_ingestion = DataIngestion()
    unscaled_dataset_path = data_ingestion.initiate_data_ingestion()


    stage1_data_transformation = Stage1DataTransformation()
    scaled_df = stage1_data_transformation.get_scaled_data(unscaled_dataset_path)


    stage2_data_transformation = Stage2DataTransformation()
    undersample_df, test_x, test_y = stage2_data_transformation.undersample_splitting(scaled_df)
    undersample_df_after_outlier_removal = stage2_data_transformation.removing_outlier(undersample_df)


    stage1_model_training = Stage1ModelTraining()
    undersample_accuracy = stage1_model_training.model_selection(undersample_df_after_outlier_removal, test_x, test_y)


    stage3_data_transformation = Stage3DataTransformation()
    oversample_df = stage3_data_transformation.oversample_splitting(scaled_df)
    oversample_df_after_outlier_removal = stage3_data_transformation.oversample_outlier_removal(oversample_df)


    stage2_model_training = Stage2ModelTraining()
    oversample_accuracy = stage2_model_training.model_selection_and_training(oversample_df_after_outlier_removal)

    accuracy_dic = {
        "Techinque":["Random undersampling","Oversampling (SMOTE)"],
        "Score": [undersample_accuracy, oversample_accuracy]
    }

    accuracy_report = pd.DataFrame(accuracy_dic)
    print(accuracy_report)