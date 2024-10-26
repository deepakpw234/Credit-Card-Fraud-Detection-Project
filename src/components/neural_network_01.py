import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from keras import Sequential
from keras.layers import Dense
from keras.activations import sigmoid
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report


@dataclass
class Stage1NeuralNetworkConfig:
    model_diectory = os.path.join(os.getcwd(),"artifacts","model")
    model_path = os.path.join(os.getcwd(),"artifacts","model","model.pkl")

    os.makedirs(model_diectory,exist_ok=True)

class Stage1NeuralNetwork:
    def __init__(self):
        self.stage1_neural_network_config = Stage1NeuralNetworkConfig()


    def undersample_neural_network_training(self,undersample_df_outlier_removed,scaled_df):
        try:
            logging.info("Stage 1 Neural network training is started")
            X = undersample_df_outlier_removed.drop("Class",axis=1)
            y = undersample_df_outlier_removed["Class"]

            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
            logging.info("train test split done")

            # Converting into Array
            X_train = X_train.values
            X_test = X_test.values
            y_train = y_train.values
            y_test = y_test.values

            logging.info("Neural network layering is started")
            # Training neural network for undersample
            input_neurons = X_train.shape[1]
            undersample_model = Sequential([Dense(input_neurons,input_dim = input_neurons,activation="relu"),
                                Dense(32,activation = "relu"),
                                Dense(1,activation="sigmoid")])
            

            undersample_model.compile(optimizer=Adam(learning_rate=0.001),loss="binary_crossentropy",metrics=["accuracy"])
            

            undersample_model.fit(X_train,y_train,batch_size=20,epochs=20,shuffle=True,verbose=0)
            logging.info("Neural network training is completed")

            logging.info("Confusion matrix and classification report calculated for undersample neural network")
            undersample_pred = undersample_model.predict(X_test,batch_size=20)
            undersample_pred = np.where(undersample_pred > 0.5, 1,0)
            print(f"Neural Network confusion matrix for undersample with undersample dataset: \n{confusion_matrix(y_test,undersample_pred)}")
            print(f"Neural Network classification report for undersample with undersample dataset: \n{classification_report(y_test,undersample_pred)}")
            print("="*60)


            # Testing on our actual original data
            original_X = scaled_df.drop("Class",axis=1)
            original_y = scaled_df["Class"]

            original_X = original_X.values
            original_y = original_y.values

            original_y_pred = undersample_model.predict(original_X,batch_size=300,verbose=0)
            original_y_pred = np.where(original_y_pred > 0.5, 1,0)
            logging.info("Confusion matrix and classification report calculated for actual sample neural network")

            print(f"Neural Network confusion matrix for Under sample with original datset: \n{confusion_matrix(original_y,original_y_pred)}")
            print(f"Neural Network classification report for Under sample with original datset: \n{classification_report(original_y,original_y_pred)}")
            print("="*60)

            logging.info("Stage 1 neural network training is completed")

        except Exception as e:
            raise CustomException(e,sys)