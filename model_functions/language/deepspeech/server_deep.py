import os
from tempfile import NamedTemporaryFile

import torch
from flask import Flask, request, jsonify
import logging
from data.data_loader import SpectrogramParser
from decoder import GreedyDecoder
from opts import add_decoder_args, add_inference_args
from transcribe import transcribe
from utils import load_model

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['.wav', '.mp3', '.ogg', '.webm'])


@app.before_first_request
def build():
    global model, spect_parser, decoder, device

    logging.getLogger().setLevel(logging.DEBUG)

    logging.info('Setting up server...')
    torch.set_grad_enabled(False)
    device = torch.device("cpu")
    model = load_model(device, "/workspace/models/deepspeech_final.pth", False)

    decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))

    spect_parser = SpectrogramParser(model.audio_conf, normalize=True)
    logging.info('Server initialised')
    #app.run(host="0.0.0.0", port="5005", debug=True, use_reloader=False)


@app.route('/transcribe', methods=['POST'])
def transcribe_file():
    if request.method == 'POST':
        res = {}
        print(request)
        if request:
            print(request.data)
        print(request.files)
        print(request.args)
        print(request.form)
        print(request.values)
        if 'file' not in request.files:
            res['status'] = "error"
            res['message'] = "audio file should be passed for the transcription"
            return jsonify(res)
        file = request.files['file']
        filename = file.filename
        _, file_extension = os.path.splitext(filename)
        if file_extension.lower() not in ALLOWED_EXTENSIONS:
            res['status'] = "error"
            res['message'] = "{} is not supported format.".format(file_extension)
            return jsonify(res)
        with NamedTemporaryFile(suffix=file_extension) as tmp_saved_audio_file:
            file.save(tmp_saved_audio_file.name)
            logging.info('Transcribing file...')
            transcription, _ = transcribe(audio_path=tmp_saved_audio_file,
                                          spect_parser=spect_parser,
                                          model=model,
                                          decoder=decoder,
                                          device=device,
                                          use_half=False)
            logging.info('File transcribed')
            res['status'] = "OK"
            res['transcription'] = transcription
            return jsonify(res)


if __name__ == "__main__":
    main()
