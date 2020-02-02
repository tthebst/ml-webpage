

pgan_model = None


def pgan():
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
    global pgan_model

    print(torch.__version__)
    if not pgan_model:
        # download model from GCS
        use_gpu = True if torch.cuda.is_available() else False
        model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                               'PGAN', model_name='celebAHQ-512',
                               pretrained=True, useGPU=use_gpu)
    model.eval()

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
    """
    url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "/tmp/dog.jpg")
    try:
        urllib.request.urlretrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    """

    # prepare data

    num_images = 1
    noise, _ = pgan_model.buildNoiseData(num_images)
    with torch.no_grad():
        generated_images = pgan_model.test(noise)

    # let's plot these images using torchvision and matplotlib
    import matplotlib.pyplot as plt
    import torchvision
    grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
    plt.imsave("/tmp/generated.jpeg", grid.permute(1, 2, 0).cpu().numpy())

    with open("/tmp/generated.jpeg", rb) as f:
        img_str = base64.b64encode(f.read())
    print(type(img_str))
    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    print("done sending response")

    return (jsonify(str(img_str)), 200, headers)
