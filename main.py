from src.components.data_ingestion import DataIngestion
from src.components.stage_01_data_transformation import Stage1DataTransformation







if __name__=="__main__":
    data_ingestion = DataIngestion()
    unscaled_dataset_path = data_ingestion.initiate_data_ingestion()


    a = Stage1DataTransformation()
    a.get_scaled_data(unscaled_dataset_path)