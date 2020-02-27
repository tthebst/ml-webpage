import time
from flask import Flask, request
from flask import render_template
import os
from flask import jsonify
import json
import sys
import servingmodels
import tempfile
import soundfile as sf
import scipy.signal
import requests
import scipy.io.wavfile
import logging
import docker

app = Flask(__name__)

"""

@app.before_request
def load_models():
    import torch
    global en2de_model
    global en2fr_model
    en2de_model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
    en2fr_model = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt')
"""


@app.route('/')
def home():

    return "Nothing to see here...."


@app.route('/en2de', methods=["GET", "POST"])
def en2de():

    try:

        print(type(request.data.decode()))
        to_pred = json.loads(request.data.decode())
        print("to predict", to_pred)
        to_send = servingmodels.en2de(request, to_pred=str(to_pred['a']))
        print(to_send)

    except Exception as e:
        logger.exception(e)
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
        print("error")
        return ("ERROR OCCURED", 404, headers)
    return to_send


@app.route('/en2fr', methods=["GET", "POST"])
def en2fr():

    try:
        print(type(request.data.decode()))
        to_pred = json.loads(request.data.decode())
        print("to predict", to_pred)
        to_send = servingmodels.en2fr(request, to_pred=str(to_pred['a']))
        print(to_send)

    except Exception as e:
        logger.exception(e)
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
        print("error")
        return ("ERROR OCCURED", 404, headers)
    return to_send


@app.route('/deepspeech_transcribe', methods=["GET", "POST", "OPTIONS"])
def deepspeech():

    # CORS RESPONSE
    if request.method == 'OPTIONS':
        # Allows GET requests from origin https://mydomain.com with
        # Authorization header
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Authorization',
            'Access-Control-Max-Age': '3600',
            'Access-Control-Allow-Credentials': 'true'
        }
        return ('', 204, headers)

    try:

        data = request.files['data'].read()

        with tempfile.NamedTemporaryFile(mode='wb') as tmp:
            print(tmp.name)
            tmp.write(data)
            print("written binary data to temporary file")
            data, samplerate = sf.read(tmp.name)

        new_rate = 16000
        number_of_samples = round(len(data) * float(new_rate) / samplerate)
        data_16000 = scipy.signal.resample(data, number_of_samples)

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.wav') as tmp:
            scipy.io.wavfile.write(tmp.name, new_rate, data_16000)
            print("written resampled binary data to temporary file")
            data, samplerate = sf.read(tmp.name)
            payload = {'file_id': '1234'}
            # client = docker.DockerClient()
            # container = client.containers.get("deepspeech")
            # ip_add = container.attrs['NetworkSettings']['IPAddress']
            resp = requests.post("http://172.20.0.5:5005/transcribe", files={'file': open(tmp.name, 'rb')}, verify=False)

        transcript = json.loads(resp.text)["transcription"]

    except Exception as e:
        print(e)
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
        print("error")
        return ("ERROR OCCURED", 404, headers)

    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    return (jsonify(str(transcript)), 200, headers)


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    app.run(port=5003, debug=True, host="0.0.0.0",  threaded=True)
