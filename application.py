import os
import sys

from flask import Flask, render_template,redirect,request,url_for

from src.logger import logging
from src.exception import CustomException

application = Flask(__name__)

app = application


@app.route("/",method=["GET","POST"])
def index():
    return render_template("index.html")

@app.route("/home",method = ["GET","POST"])
def home():
    try:
        if request.method == "GET":
            render_template("index.html")

        else:
            pass


        
    except Exception as e:
        raise CustomException(e,sys)




if __name__=="__main__":
    app.run(debug =True)