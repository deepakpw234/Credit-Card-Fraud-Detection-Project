import os
import sys

from flask import Flask, render_template,redirect,request,url_for

from src.logger import logging
from src.exception import CustomException

application = Flask(__name__)

app = application


@app.route("/",methods=["GET","POST"])
def index():
    return render_template("index.html")

@app.route("/home",methods = ["GET","POST"])
def predict_details():
    try:
        if request.method == "GET":
            render_template("index.html")

        else:
            data = request.form.get("age")

            print(data)

            render_template("home.html",data=data)


    except Exception as e:
        raise CustomException(e,sys)

    return render_template("index.html")
    




if __name__=="__main__":
    app.run(debug =True)