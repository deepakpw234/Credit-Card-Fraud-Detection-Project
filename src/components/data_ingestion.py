import os
import sys
from dataclasses import dataclass
import urllib
import urllib.request
import zipfile

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    dataset_path = os.path.join(os.getcwd(),"artifacts")
    dataset_zip_path = os.path.join(os.getcwd(),"artifacts","creditcard.zip")
    dataset_file_path = os.path.join(os.getcwd(),"artifacts","creditcard.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion is started")
            logging.info("Downloading of dataset from github is started")

            github_url = "https://github.com/deepakpw234/Project-Datasets/raw/refs/heads/main/creditcard.zip"

            urllib.request.urlretrieve(github_url,self.data_ingestion_config.dataset_zip_path)
            
            logging.info("Dataset unzipping is started")

            with zipfile.ZipFile(os.path.join(self.data_ingestion_config.dataset_zip_path),"r") as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.dataset_path)

            logging.info("Dataset is unzipped")

            logging.info("Data Ingestion is completed")

        except Exception as e:
            raise CustomException(e,sys)
        
        return self.data_ingestion_config.dataset_file_path
    



    