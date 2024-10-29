import os
import sys
import pandas as pd
import numpy as np
import random
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


@dataclass
class PredictionPipelineConfig:
    model_path = os.path.join(os.getcwd(),"artifacts","model","model.pkl")


class PredictionPipeline:
    def __init__(self):
        self.prediction_pipeline_config = PredictionPipelineConfig()

    def random_number_generation(self, random_number):
        try:

            logging.info("Prediction pipleline is started")
            min_num = [-0.307413,-0.994983,-56.4075,-72.7157,-48.3255,-5.6831,-113.7433,-26.1605,-43.5572,-73.2167,-13.4340,-24.5882,-4.7974,-18.6837,-5.7918,-19.2143,-4.4989,-14.1298,-25.1628,-9.4987,-7.2135,-54.4977,-34.8303,-10.9331,-44.8077,-2.8366,-10.2954,-2.6045,-22.5656,-15.4300]
            max_num = [358.683155,1.035022,2.4549,22.0577,9.3825,16.8753,34.8016,73.3016,120.5895,20.0072,15.5949,23.7451,12.0189,7.8483,7.1268,10.5267,8.8777,17.3151,9.2535,5.0410,5.5919,39.4209,27.2028,10.5030,22.5284,4.5845,7.5195,3.5173,31.6122,33.8478]


            random_value_for_prediction = {}

            for i in range(30):
                if i == 0:
                    temp = round(random.uniform(min_num[i],max_num[i]),2)
                    random_value_for_prediction["Amount_scale"]=temp

                elif i == 1:
                    temp = round(random.uniform(min_num[i],max_num[i]),2)
                    random_value_for_prediction["Time_scale"]=temp
                
                else:
                    temp = round(random.uniform(min_num[i],max_num[i]),2)
                    random_value_for_prediction[f"V{i-1}"]=temp

            logging.info("Random numbers are generated for all required variable")

            # print(random_value_for_prediction)

        except Exception as e:
            raise CustomException(e,sys)
        
        return random_value_for_prediction
    

    def fraud_prediction(self, random_value_for_prediction):
        try:
            logging.info("Fraud detection from the required input is started")
            df = pd.DataFrame(random_value_for_prediction, index=[0])

            df = df.values

            logging.info("Loading the model")
            model = load_object(self.prediction_pipeline_config.model_path)
            
            result_predict = model.predict(df)

            logging.info("Predicting the transaction")
            if result_predict[0][0] < 0.5:
                prediction = "This is a legit transaction"
            
            else:
                prediction = "This is a fraud transaction"
                

        except Exception as e:
            raise CustomException(e,sys)
        
        return prediction
        



        
