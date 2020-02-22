import time
from flask import Flask, request
from flask import render_template
import os
import json
import sys
import servingmodels
import tempfile
import soundfile as sf
import scipy.signal
import requests
import scipy.io.wavfile

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
    print(request.files)
    print('data' in request.files)

    data = request.files['data'].read()

    with tempfile.NamedTemporaryFile(mode='wb') as tmp:
        print(tmp.name)
        tmp.write(data)
        print("written binary data to temporary file")
        data, samplerate = sf.read(tmp.name)

    print(data, samplerate)
    new_rate = 16000
    number_of_samples = round(len(data) * float(new_rate) / samplerate)
    data_16000 = scipy.signal.resample(data, number_of_samples)
    print("resampled data")

    with tempfile.NamedTemporaryFile(mode='wb', suffix='.wav') as tmp:
        scipy.io.wavfile.write(tmp.name, new_rate, data_16000)
        print("written resampled binary data to temporary file")
        data, samplerate = sf.read(tmp.name)
        payload = {'file_id': '1234'}
        print(data, samplerate)
        print(tmp.name)
        resp = requests.post("http://172.18.0.2:5005/transcribe", files={'file': open(tmp.name, 'rb')}, verify=False)
        print(resp.request.body)

        print(resp.request.headers)
        time.sleep(1)
        print(resp.text)

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
