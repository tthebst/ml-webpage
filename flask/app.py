from flask import Flask, request
from flask import render_template
import os
app = Flask(__name__)


@app.route('/')
def home():

    return render_template('home.html')


@app.route('/classification', methods=["GET", "POST"])
def classification():

    if request.method == "POST":

        imfile = request.files['photo']

        imfile.save("hallo.jpg")

    return render_template('classification.html')
