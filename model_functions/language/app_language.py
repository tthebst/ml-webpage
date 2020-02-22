import time
from flask import Flask, request
from flask import render_template
import os
import json
import sys
import servingmodels
import soundfile as sf

app = Flask(__name__)


@app.route('/')
def home():

    return "Nothing to see here...."


@app.route('/en2de', methods=["GET", "POST"])
def en2de():

    try:
        to_pred = json.loads(request.data.decode())
        print("to predict", to_pred)
        to_send = servingmodels.en2de(request, to_pred=str(to_pred['a']))
        print(to_send)

    except Exception as e:
        print(e)
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
        print("error")
        return ("ERROR OCCURED", 404, headers)
    return to_send


@app.route('/deepspeech_transcribe', methods=["GET", "POST", "OPTIONS"])
def deepspeech():
    print("HIU")
    print(request.form)
    print(request.files)
    print(request.files['data'])
    data = request.files['data'].read()
    print(data)
    with open("/tmp/sound1.ogg", 'wb') as f:
        f.write(data)

    data, samplerate = sf.read('/tmp/sound1.ogg')
    print(data, samplerate)
    try:
        to_pred = json.loads(request.data.decode())
        print("to predict", to_pred)
        to_send = servingmodels.en2de(request, to_pred=to_pred['a'])
        print(to_send)

    except Exception as e:
        print(e)
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
        print("error")
        return ("ERROR OCCURED", 404, headers)
    return to_send


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(port=5003, debug=True, host="0.0.0.0",  threaded=False)
