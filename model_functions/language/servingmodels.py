

en2de = None


def pgan(request):
    """Responds to any HTTP request.
   Args:
        request(flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response < http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """

    # imports
    import pickle as pk
    from flask import jsonify
    import urllib.request
    from PIL import Image
    import torch
    import json
    import requests
    import base64
    from io import BytesIO

    # download & prepare model if necessary
    global en2de

    print(torch.__version__)
    # Download and save model
    if not en2de:
        # download model from GCS
        use_gpu = True if torch.cuda.is_available() else False
        en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')

    # answers for CORS request
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

    de = en2de.translate('PyTorch Hub is a pre-trained model repository designed to facilitate research reproducibility.')

    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    print("done sending response")

    return (jsonify(str(img_str)), 200, headers)
