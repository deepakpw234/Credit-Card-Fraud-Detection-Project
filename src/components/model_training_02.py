import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split ,cross_val_predict
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report
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
            oversample_df = smote_df

            Xsm = oversample_df.drop("Class",axis=1)
            ysm = oversample_df[["Class"]]

            print(Xsm.shape)
            print(ysm.shape)

            Xsm_train, Xsm_test, ysm_train, ysm_test = train_test_split(Xsm, ysm, test_size=0.2,random_state=42)

            print(Xsm_train.shape)
            print(ysm_test.value_counts())


            log_params ={
                "penalty": ['l2'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }
            log_reg_random = RandomizedSearchCV(LogisticRegression(solver="lbfgs",max_iter=1000),param_distributions=log_params)
            log_reg_random.fit(Xsm_train,ysm_train)
            log_reg = log_reg_random.best_estimator_


            # Checking for ROC AUC Score
            log_reg_random_cross_validation_predict = cross_val_predict(log_reg,Xsm_train,ysm_train,cv=5,method="decision_function")

            print(f"Logistic Regression ROC AUC score is: {roc_auc_score(ysm_train,log_reg_random_cross_validation_predict)}")

        except Exception as e:
            raise CustomException(e,sys)