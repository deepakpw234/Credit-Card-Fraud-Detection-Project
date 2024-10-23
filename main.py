from src.components.data_ingestion import DataIngestion
from src.components.stage_01_data_transformation import Stage1DataTransformation
from src.components.stage_02_data_transformation import Stage2DataTransformation
from src.components.model_training_01 import Stage1ModelTraining






if __name__=="__main__":
    data_ingestion = DataIngestion()
    unscaled_dataset_path = data_ingestion.initiate_data_ingestion()


    stage1_data_transformation = Stage1DataTransformation()
    scaled_df = stage1_data_transformation.get_scaled_data(unscaled_dataset_path)


    stage2_data_transformation = Stage2DataTransformation()
    undersample_df, test_x, test_y = stage2_data_transformation.undersample_splitting(scaled_df)
    undersample_df_after_outlier_removal = stage2_data_transformation.removing_outlier(undersample_df)


    stage1_model_training = Stage1ModelTraining()
    stage1_model_training.model_selection(undersample_df_after_outlier_removal, test_x, test_y)