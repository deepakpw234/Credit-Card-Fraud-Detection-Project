import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split ,cross_val_predict
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression

@dataclass
class Stage2ModelTrainingConfig:
    model_path = os.path.join(os.getcwd(),"artifacts","model","model.pkl")
    model_directory = os.path.join(os.getcwd(),"artifacts","model")

    os.makedirs(model_directory,exist_ok=True)


class Stage2ModelTraining:
    def __init__(self):
        self.stage2_model_training_config = Stage2ModelTrainingConfig()


    def model_selection_and_training(self, smote_df):
        try:
            logging.info("Model selection and training is started for oversample")
            oversample_df = smote_df

            Xsm = oversample_df.drop("Class",axis=1)
            ysm = oversample_df["Class"]


            Xsm_train, Xsm_test, ysm_train, ysm_test = train_test_split(Xsm, ysm, test_size=0.2,random_state=42)
            logging.info("Train test split is completed for undersample data")


            Xsm_train = Xsm_train.values
            Xsm_test = Xsm_test.values
            ysm_train = ysm_train.values
            ysm_test = ysm_test.values


            log_params ={
                "penalty": ['l2'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }
            log_reg_random = RandomizedSearchCV(LogisticRegression(),param_distributions=log_params, n_iter=7)
            log_reg_random.fit(Xsm_train,ysm_train)
            log_reg = log_reg_random.best_estimator_

            logging.info("Best parameters for model is calculated on oversample data")


            # Checking for ROC AUC Score
            log_reg_random_cross_validation_predict = cross_val_predict(log_reg,Xsm_train,ysm_train,cv=5,method="decision_function")

            # print(f"Logistic Regression ROC AUC score is: {roc_auc_score(ysm_train,log_reg_random_cross_validation_predict)}")

            logging.info("Checking for accuracy score, confusion matrix and classification report for oversample on training data")
            print("="*60)
            print("Result for oversample Xsm_train")
            oversample_y_pred = log_reg.predict(Xsm_train)
            print(f"Accuracy: {accuracy_score(ysm_train,oversample_y_pred)}")
            print(f"Confusion matrix: \n{confusion_matrix(ysm_train,oversample_y_pred)}")
            print(f"classification report: \n{classification_report(ysm_train,oversample_y_pred)}")
            print("="*60)

            logging.info("Checking for accuracy score, confusion matrix and classification report for oversample on test data")
            print("Result for oversample Xsm_test")
            oversample_y_pred = log_reg.predict(Xsm_test)
            oversample_accuracy = accuracy_score(ysm_test,oversample_y_pred)
            print(f"Accuracy: {oversample_accuracy}")
            print(f"Confusion matrix: \n{confusion_matrix(ysm_test,oversample_y_pred)}")
            print(f"classification report: \n{classification_report(ysm_test,oversample_y_pred)}")
            print("="*60)

        except Exception as e:
            raise CustomException(e,sys)
        
        return oversample_accuracy