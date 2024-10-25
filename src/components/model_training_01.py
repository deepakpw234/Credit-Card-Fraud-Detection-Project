import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score, roc_auc_score,classification_report


@dataclass
class Stage1ModelTrainingConfig:
    model_diectory = os.path.join(os.getcwd(),"artifacts","model")
    model_path = os.path.join(os.getcwd(),"artifacts","model","model.pkl")

    os.makedirs(model_diectory,exist_ok=True)


class Stage1ModelTraining:
    def __init__(self):
        self.stage1_model_training = Stage1ModelTrainingConfig()

    def model_selection(self,undersample_df_outlier_removed, test_x, test_y):
        try:
            X = undersample_df_outlier_removed.drop("Class",axis=1)
            y = undersample_df_outlier_removed["Class"]

            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
            # print(X_train)
            # print(y_test)


            # Converting into Array
            X_train = X_train.values
            X_test = X_test.values
            y_train = y_train.values
            y_test = y_test.values



            models = {
                "Logistic Regression": LogisticRegression(),
                "KNeighbors Classifier" : KNeighborsClassifier(),
                "Support Vector Classifier": SVC(),
                "DecisionTree Classifier":DecisionTreeClassifier()
            }

            # Checking for model cross validation score
            for i in range(len(models.keys())):
                model = list(models.values())[i]
                training_score = cross_val_score(model,X_train,y_train,cv=5)
                print(f"Classifier: {list(models.keys())[i]} has the cross validation score of {round(training_score.mean()*100,2)}%")

            '''
            Classifier: Logistic Regression has the cross validation score of 94.15%
            Classifier: KNeighbors Classifier has the cross validation score of 93.41%
            Classifier: Support Vector Classifier has the cross validation score of 92.36%
            Classifier: DecisionTree Classifier has the cross validation score of 88.8%
            '''

            # Hyperparameter tunning for all the classifier
            # Finding the Best parameter
            log_params ={
                "penalty": ['l2'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }
            log_reg_grid = GridSearchCV(LogisticRegression(solver="lbfgs",max_iter=1000),param_grid=log_params)
            log_reg_grid.fit(X_train,y_train)
            log_reg = log_reg_grid.best_estimator_


            knear_params = {
                "n_neighbors": list(range(2,5,1)), 
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'] 
            }
            knear_grid = GridSearchCV(KNeighborsClassifier(),param_grid=knear_params)
            knear_grid.fit(X_train,y_train)
            knear_clf = knear_grid.best_estimator_


            svm_params = {
                'C': [0.5, 0.7, 0.9, 1], 
                'kernel': ['rbf', 'poly', 'sigmoid', 'linear']
            }
            svm_grid = GridSearchCV(SVC(),param_grid=svm_params)
            svm_grid.fit(X_train,y_train)
            svm_clf = svm_grid.best_estimator_


            tree_params = {
                "criterion":["gini", "entropy"],
                "splitter":['best','random'],
                'max_depth':[3,4,5,6],
                'min_samples_split':list(range(8, 20, 2)),
                'min_samples_leaf':[5,6,7],
            }
            tree_grid = GridSearchCV(DecisionTreeClassifier(),param_grid=tree_params)
            tree_grid.fit(X_train,y_train)
            tree_clf = tree_grid.best_estimator_

            # print(log_reg)
            # print(knear_clf)
            # print(svm_clf)
            # print(tree_clf)

            # Checking for ROC AUC Score
            log_reg_cross_validation_predict = cross_val_predict(log_reg,X_train,y_train,cv=5,method="decision_function")
            knear_validation_predict = cross_val_predict(knear_clf,X_train,y_train,cv=5)
            svm_validation_predict = cross_val_predict(svm_clf,X_train,y_train,cv=5,method="decision_function")
            tree_validation_predict = cross_val_predict(tree_clf,X_train,y_train,cv=5)

            print(f"Logistic Regression ROC AUC score is: {roc_auc_score(y_train,log_reg_cross_validation_predict)}")
            print(f"Knear ROC AUC score is: {roc_auc_score(y_train,knear_validation_predict)}")
            print(f"SVM ROC AUC score is: {roc_auc_score(y_train,svm_validation_predict)}")
            print(f"Decision Tree ROC AUC score is: {roc_auc_score(y_train,tree_validation_predict)}")


            '''
            Logistic Regression ROC AUC score is: 0.972593149540518
            Knear ROC AUC score is: 0.9284565580618213
            SVM ROC AUC score is: 0.9706329713171818
            Decision Tree ROC AUC score is: 0.9142648287385131
            '''


            '''
            From the above two checking points (Accuracy score and ROC AUC score) 
            we are selecting Logistic regression for futher model fitting
            '''

            print("Result for undersample X_train")
            undersample_y_pred = log_reg.predict(X_train)
            print(f"Accuracy: {accuracy_score(y_train,undersample_y_pred)}")
            print(f"Confusion matrix: \n{confusion_matrix(y_train,undersample_y_pred)}")
            print(f"classification report: \n{classification_report(y_train,undersample_y_pred)}")
            print("="*60)


            print("Result for undersample X_test")
            undersample_y_pred = log_reg.predict(X_test)
            undersample_accuracy = accuracy_score(y_test,undersample_y_pred)
            print(f"Accuracy: {undersample_accuracy}")
            print(f"Confusion matrix: \n{confusion_matrix(y_test,undersample_y_pred)}")
            print(f"Classification report: \n{classification_report(y_test,undersample_y_pred)}")
            print("="*60)


            print("Result for undersample 40000 sample")
            testing = log_reg.predict(test_x.values)
            print(len(testing))
            print(f"Accuracy: {accuracy_score(test_y,testing)}")
            print(f"Confusion matrix: \n{confusion_matrix(test_y,testing)}")
            print(f"Classification report: \n{classification_report(test_y,testing)}")
            print("="*60)


        except Exception as e:
            raise CustomException(e,sys)
        
        return undersample_accuracy
