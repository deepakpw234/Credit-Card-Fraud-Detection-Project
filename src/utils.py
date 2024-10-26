import os
import sys
import dill

from src.exception import CustomException
from src.logger import logging


def save_object(model_path,obj):
    try:
        
        with open(model_path,"wb") as file_obj:
            dill.dump(obj, file_obj) 

        logging.info("Model is saved in binary format in pikel file")

    except Exception as e:
        raise CustomException(e,sys)