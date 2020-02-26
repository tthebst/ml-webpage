from flask import jsonify
from flask import Flask, request
from flask import render_template
import requests
import os
import scipy
import urllib.request
import json
import tempfile
import soundfile as sf
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


@app.route('/language/deepspeech', methods=["GET", "POST"])
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

            resp = requests.post("http://172.18.0.3:5005/transcribe", files={'file': open(tmp.name, 'rb')}, verify=False)

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


@app.route('/generative/biggan', methods=["GET", "POST"])
def generative_biggan():
    print("got post to bgigan")
    print(request.data)
    resp = requests.post("http://172.18.0.4:5002/biggan", data=request.data, verify=False)
    print(resp)

    print(resp.json())
    return resp


if __name__ == '__main__':
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(port=5000, debug=True, host="0.0.0.0")
