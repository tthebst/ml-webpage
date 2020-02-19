import time
from flask import Flask, request
from flask import render_template
import os
import tensorflow_hub as hub
import tensorflow as tf
import json
import sys
import servingmodels
from tensorflow import keras
app = Flask(__name__)

module = None
sess = None
graph = None
output = None
inputs = None


@app.route('/')
def home():

    return "Nothing to see here...."


@app.route('/en2de', methods=["GET", "POST"])
def en2de():

    try:
        to_pred = json.loads(request.data.decode())
        print("to predict", to_pred)
        to_send = servingmodels.en2de(request, to_pred=str(to_pred['a']))

    except Exception as e:
        print(e)
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
        print("error")
        return ("ERROR OCCURED", 404, headers)
    return to_send


if __name__ == '__main__':
    app.run(port=5003, debug=True, host="0.0.0.0",  threaded=False)
