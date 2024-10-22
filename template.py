import os
from pathlib import Path

project_name = "Credit_Card_Fraud_Detection"

list_of_files = [
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_training.py",
    "src/pipelines/__init__.py",
    "src/pipelines/train_pipeline.py",
    "src/pipelines/predict_pipeline.py",
    "src/logger.py",
    "src/exception.py",
    "src/utils.py",
    "notebook/abc.ipynb",
    "artifacts/abc.text",
    "main.py",
    "application.py",
    "requirements.txt",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    # print(filepath)

    file_dic, filename = os.path.split(filepath)
    print(file_dic)
    print(filename)
    if file_dic != "":
        os.makedirs(file_dic,exist_ok=True)


    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, "w") as f:
            pass

