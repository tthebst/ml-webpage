from flask import Flask, request
from flask import render_template
import os
import urllib.request
import json
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


@app.route('/object_detect', methods=["GET", "POST"])
def object_detect():

    return render_template('object_detect.html')


@app.route('/language', methods=["GET", "POST"])
def language():

    return render_template('language.html')


@app.route('/generative', methods=["GET", "POST"])
def generative():

    class_idx = json.loads(urllib.request.urlopen("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json").read())
    label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    idx = [k for k in range(len(class_idx))]

    return render_template('generative.html', label=label, idx=idx)


if __name__ == '__main__':
    app.run(port=5000, debug=True, host="0.0.0.0")
