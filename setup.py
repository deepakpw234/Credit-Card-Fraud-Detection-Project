from setuptools import find_packages,setup
import os
import sys

HYPHEN_DOT_E = "-e ."
def get_requirements_details(filename):
    with open(filename,'r') as file:
        requirements = file.read().splitlines()
        
    if HYPHEN_DOT_E in requirements:

        requirements.remove(HYPHEN_DOT_E)

    return requirements


setup(
    name = "Credit Card Fraud Detection App",
    version="0.0.0.1",
    description="This app is created to detect fraud transaction of credit card",
    long_description= "This app helps the credit card company to detect the fraud transaction and block it for futher use. This app uses the machine learning model to predict the transaction category",
    author = "Deepak Pawar",
    author_email="deepakpe234@gmail.com",
    packages = find_packages(),
    install_requirements = get_requirements_details('requirements.txt')
)

