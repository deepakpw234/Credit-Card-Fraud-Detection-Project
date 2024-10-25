import os
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
class Stage2NeuralNetworkConfig:
    model_diectory = os.path.join(os.getcwd(),"artifacts","model")
    model_path = os.path.join(os.getcwd(),"artifacts","model","model.pkl")

    os.makedirs(model_diectory,exist_ok=True)



class Stage2NeuralNetwork:
    def __init__(self):
        stage2_neural_network_config = Stage2NeuralNetworkConfig()


    def oversample_neural_network_training(self,oversample_df_after_outlier_removal,scaled_df):
        try:
            X = oversample_df_after_outlier_removal.drop("Class",axis=1)
            y = oversample_df_after_outlier_removal["Class"]

            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
            # print(X_train)
            # print(y_test)


            # Converting into Array
            X_train = X_train.values
            X_test = X_test.values
            y_train = y_train.values
            y_test = y_test.values

            # Making neural network for oversample 
            input_neuron = X_train.shape[1]
            oversample_model = Sequential([Dense(input_neuron,input_dim = input_neuron, activation="relu"),
                                           Dense(32,activation="relu"),
                                           Dense(32,activation="relu"),
                                           Dense(1,activation="sigmoid")])
            
            oversample_model.compile(optimizer=Adam(learning_rate=0.001),loss="binary_crossentropy",metrics=["accuracy"])

            oversample_model.fit(X_train,y_train,batch_size=500,epochs=50,verbose=2,shuffle=True)

            oversample_y_pred = oversample_model.predict(X_test,batch_size=500)
            oversample_y_pred = np.where(oversample_y_pred > 0.5, 1,0)

            print(f"Neural Network confusion matrix for oversample: \n{confusion_matrix(y_test,oversample_y_pred)}")
            print(f"Neural Network classification report for oversample: \n{classification_report(y_test,oversample_y_pred)}")
            print("="*60)


            # Testing on our actual original data
            original_X = scaled_df.drop("Class",axis=1)
            original_y = scaled_df["Class"]

            original_X = original_X.values
            original_y = original_y.values

            original_y_pred = oversample_model.predict(original_X,batch_size=500,verbose=2)
            original_y_pred = np.where(original_y_pred > 0.5, 1,0)

            print(f"Neural Network confusion matrix for original sample: \n{confusion_matrix(original_y,original_y_pred)}")
            print(f"Neural Network classification report for original sample: \n{classification_report(original_y,original_y_pred)}")
            print("="*60)



        except Exception as e:
            raise CustomException(e,sys)