from flask import Flask, request
from flask import render_template
import os
import json
import models
app = Flask(__name__)


@app.route('/')
def home():

    return "Nothing to see here...."


@app.route('/deeplabv3', methods=["POST"])
def classification():

    if request.method == "POST":
        imfile = request.files['photo']
        imfile.save("/tmp/to_pred.jpg")

        to_send = models.deeplabv3(request)
        print(to_send)

    return to_send


if __name__ == '__main__':
    app.run(port=5001, debug=True)
