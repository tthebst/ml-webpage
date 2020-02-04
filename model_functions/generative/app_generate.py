from flask import Flask, request
from flask import render_template
import os
import tensorflow_hub as hub
import tensorflow as tf
import json
import models
from tensorflow import keras
app = Flask(__name__)


@app.route('/')
def home():

    return "Nothing to see here...."


@app.route('/pgan', methods=["GET"])
def pgan():

    try:
        to_send = models.pgan()

    except Exception as e:
        print(e)
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
        print("error")
        return ("ERROR OCCURED", 404, headers)
    return to_send


@app.route('/biggan', methods=["GET", "POST"])
def biggan():
    to_send = models.biggan(request)
    try:
        pass
    except Exception as e:
        print(e)
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
        print("error")
        return ("ERROR OCCURED", 404, headers)
    return to_send


if __name__ == '__main__':
    app.run(port=5002, debug=True, host="0.0.0.0",  threaded=False)
