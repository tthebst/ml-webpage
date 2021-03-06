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


@app.route('/fastrnn', methods=["POST"])
def fastrnn():

    if request.method == "POST":
        imfile = request.files['photo']
        print(imfile)
        imfile.save("/tmp/to_pred.jpg")
        print("saved image")
        to_send = models.fastrnn(request)
        print(to_send)

    return to_send


@app.route('/fastrnn2', methods=["POST"])
def fastrnn2():

    if request.method == "POST":
        imfile = request.files['photo']
        print(imfile)
        imfile.save("/tmp/to_pred.jpg")
        print("saved image")
        to_send = models.fastrnn2(request)
        print(to_send)

    return to_send


if __name__ == '__main__':
    app.run(port=5001, debug=True, host="0.0.0.0")
