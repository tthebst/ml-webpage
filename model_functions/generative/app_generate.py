from flask import Flask, request
from flask import render_template
import os
import json
import models
from tensorflow import keras
app = Flask(__name__)


@app.route('/')
def home():

    return "Nothing to see here...."


@app.route('/pgan', methods=["GET"])
def pgan():

    to_send = models.pgan()

    return to_send


if __name__ == '__main__':
    app.run(port=5002, debug=True, host="0.0.0.0")
