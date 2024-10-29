import os
import sys

from flask import Flask, render_template,redirect,request,url_for

from src.logger import logging
from src.exception import CustomException

from src.pipelines.predict_pipeline import PredictionPipeline

application = Flask(__name__)

app = application


@app.route("/",methods=["GET","POST"])
def index():
    return render_template("index.html")

@app.route("/home",methods = ["GET","POST"])
def predict_details():
    try:
        if request.method == "GET":
            return render_template("home.html")

        else:

            logging.info("Requesting to get input from website")
            data = request.form.get("random_number")

            prediction_pipeline = PredictionPipeline()
            dict_of_all_value = prediction_pipeline.random_number_generation(data)

            final_prediction = prediction_pipeline.fraud_prediction(dict_of_all_value)

            logging.info("final predticion completed")

            print(final_prediction)

            return render_template("home.html",Amount_scale=dict_of_all_value["Amount_scale"],Time_scale=dict_of_all_value["Time_scale"],V1=dict_of_all_value["V1"],V2=dict_of_all_value["V2"],V3=dict_of_all_value["V3"],V4=dict_of_all_value["V4"],
                                   V5=dict_of_all_value["V5"],V6=dict_of_all_value["V6"],V7=dict_of_all_value["V7"],V8=dict_of_all_value["V8"],V9=dict_of_all_value["V9"],V10=dict_of_all_value["V10"],V11=dict_of_all_value["V11"],V12=dict_of_all_value["V12"],
                                   V13=dict_of_all_value["V13"],V14=dict_of_all_value["V14"],V15=dict_of_all_value["V15"],V16=dict_of_all_value["V16"],V17=dict_of_all_value["V17"],V18=dict_of_all_value["V18"],V19=dict_of_all_value["V19"],V20=dict_of_all_value["V20"],
                                   V21=dict_of_all_value["V21"],V22=dict_of_all_value["V22"],V23=dict_of_all_value["V23"],V24=dict_of_all_value["V24"],V25=dict_of_all_value["V25"],V26=dict_of_all_value["V26"],V27=dict_of_all_value["V27"],V28=dict_of_all_value["V28"],
                                   result = final_prediction)


    except Exception as e:
        raise CustomException(e,sys)



if __name__=="__main__":
    app.run(debug =True)