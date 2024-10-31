


![creditcard](https://github.com/user-attachments/assets/9210a391-cacd-47a2-859f-fe535dd05552)



# 1. Credit Card Fraud Detection Project

The project aims to detect fraud in online credit card transactions using a dataset comprising 284,807 transactions, of which 492 are fraudulent and 284,315 are legitimate. Given the significant class imbalance between fraudulent and legitimate transactions, several techniques were employed, including under-sampling, over-sampling using SMOTE (Synthetic Minority Over-sampling Technique), and a neural network model. To address this imbalance effectively, SMOTE was particularly utilized to enhance the performance of the neural network. The results were highly promising, achieving an accuracy of 99.99%. The confusion matrix for the model is as follows:
                
                    Predicted Value
                       F     L
        True    F    [489    3
        Value   L     25  284290]
- 489 - TRUE POSITIVE (model predicted fraud, actually fraud)
- 3   - FALSE NEGATIVE (model predicted legit, actually fraud)
- 25  - FALSE POSITIVE (model predicted fraud, actually legit)
- 284290 - TRUE NEGATIVE (model predicted legit, actually legit)

These results demonstrate the model's efficacy in distinguishing between fraudulent and legitimate transactions with minimal error.


# 2. Table of contents 

1. Dataset
2. Data Ingestion
3. Data Transforamtion
   - Jupyter Notebook
4. Model Training
   - Under sample training
   - Over sample (SMOTE) training
5. Neural Network
   - Under sample neural network
   - Over sample neural network
6. Prediction pipeline
7. Flask application


## 2.1 Dataset

The dataset for this project was sourced from Kaggle, comprising 284,807 transactions, with 492 labeled as fraudulent and 284,315 as legitimate. It contains 31 columns, including 'Time', 'V1' to 'V28', 'Amount', and 'Class'. Notably, the 'V1' to 'V28' columns are already scaled using Principal Component Analysis (PCA). The dataset is highly imbalanced, with fraudulent transactions accounting for only 0.17% of the total.

- You can download the dataset by [clicking here](https://github.com/deepakpw234/Project-Datasets/raw/refs/heads/main/creditcard.zip)


## 2.2 Data Ingestion

In this section of the project, the dataset was first downloaded from a GitHub repository and unzipped to prepare it for data transformation and analysis. This step ensured that the raw data was accessible and in the correct format for preprocessing. To manage the large dataset efficiently, GitHub Large File Storage (LFS) was utilized for uploading and storing.


- You can check out the data ingestion code by [clicking here](https://github.com/deepakpw234/Credit-Card-Fraud-Detection-Project/blob/main/src/components/data_ingestion.py)


## 2.3 Data Transforamtion

In this part of the project, the 'Time' and 'Amount' columns were first scaled using a robust scaler to prepare them for model training. Given the dataset's significant imbalance, specific techniques were applied to address this issue. Under-sampling and over-sampling using SMOTE (Synthetic Minority Over-sampling Technique) were employed to create new balanced sub-samples. Additionally, outliers were removed from the resampled data to enhance the model’s performance and prediction accuracy. These steps were crucial in ensuring the model's ability to accurately distinguish between fraudulent and legitimate transactions.

- You can check out the jupyter notebook by [clicking here](https://github.com/deepakpw234/Credit-Card-Fraud-Detection-Project/blob/main/notebook/Credit%20Card%20Fraud%20Detection%20Analysis%20-%20Deepak.ipynb)
- You can check out the data transforamtion code by [clicking here](https://github.com/deepakpw234/Credit-Card-Fraud-Detection-Project/blob/main/src/components/stage_01_data_transformation.py)


## 2.4 Model Training

- **Undersampling -**
Undersampling is a technique used to address class imbalance in a dataset. It involves reducing the number of instances from the majority class to make it comparable to the minority class. By doing this, the model gets an equal or more balanced representation of classes.


- **Oversampling (SMOTE) -**
Oversampling is a technique used to address class imbalance in datasets. SMOTE generates synthetic samples for the minority class by creating new instances along the line segments between the minority class instances and their nearest neighbors.

Four different models were trained and hyper-tuned, logistic regression performing the best among them. The model’s accuracy was evaluated using the ROC-AUC score and a classification report for the under-sampling and over-sampling technique.

- You can check out the model training code by [clicking here](https://github.com/deepakpw234/Credit-Card-Fraud-Detection-Project/blob/main/src/components/model_training_01.py)


## 2.5 Neural Netowrk

A simple neural network was designed with three dense layers to address both under-sampling and over-sampling techniques. Each of the hidden layers utilized the ReLU (Rectified Linear Unit) activation function, known for its efficiency in handling non-linear relationships. The output layer was structured with a single neuron and employed a sigmoid activation function, which is appropriate for binary classification tasks. This architecture enabled the model to learn and differentiate effectively between fraudulent and legitimate transactions in the dataset.

- Confusion matrix for undersample neural network:-

                [470     22
                12022  272293]

- Confusion matrix for oversample neural network:-

                [489    3
                25   284290]

The over-sampled neural network delivered impressive accuracy, making it the chosen model for predicting new entries.


## 2.6 Prediction pipeline

The training and prediction pipeline is designed to efficiently handle new transactions. In this process, each new transaction is first scaled to match the training data format. The saved and trained model is then loaded to make predictions using these scaled values. This approach ensures accurate classification of new transactions as either fraudulent or legitimate, maintaining consistency and precision in detecting fraud.


## 2.7 Flask Web Framework

The application for users was developed using the Flask web framework. This web-based interface allows users to input necessary transaction details, which are then processed by the trained model. Based on these inputs, the model predicts whether the transaction is fraudulent or legitimate.
# Project Structure

![credit card project structure](https://github.com/user-attachments/assets/0b83c814-6f79-4259-8c53-0422caccaf07)

## Installation

### Prerequisites

- pandas
- numpy
- scikit-learn
- keras == 3.5.0
- tensorflow
- ipykernel==6.29.5
- matplotlib
- seaborn
- imbalanced-learn
- dill
- Flask


### Steps

Step 1: Clone the repository
```bash
git clone https://github.com/deepakpw234/Credit-Card-Fraud-Detection-Project.git
```
Step 2: Change into the project directory
```bash
cd CAPTCHA-Break-Project
```

Step 3: Create and activate a virtual environment

```bash
conda create -n credit_env python==3.11 -y
```
```bash
conda activate credit_env
```

Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

Step 5: Run the project

```bash
python main.py
```

Step 6: Run the flask application

```bash
python application.py
```
## Live Demo

https://github.com/user-attachments/assets/2598dd1d-dcb6-4228-992d-ec4a93d8a990

# Hi, I'm Deepak Pawar! 👋


## 🚀 About Me
I'm a Data Scientist with over 2 years of experience in leveraging data analytics, statistical modelling, and machine learning to drive actionable insights and business growth. Proficient in leveraging Python, SQL, Scikit-Learn and Machine Learning techniques to solve complex data problems and enhance predictive analytics. Strong background in data preprocessing, feature engineering, and model evaluation, with a proven track record of optimizing model performance and scalability. Also, Expertise in developing and deploying end-to-end data science solutions within CI/CD pipelines, ensuring seamless integration and continuous delivery of models and applications.


## 🛠 Skills

- Languages – Python, C Programming
- Libraries – Pandas, NumPy, Scikit-Learn, TensorFlow, Keras, Transformers, Hugging face Library, Neural Netowrk
- Visualization Tools – Matplotlib, Seaborn, Power BI
- Databases – SQL, MongoDB
- Clouds – Amazon Web Service (AWS), Microsoft Azure
- Misc – GitHub Action, Docker, Flask, Jupyter Notebook, Office 365


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/deepakpw234)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/deepak-pawar-92a2a5b5/)



## Authors

- [@deepakpw234](https://github.com/deepakpw234)

