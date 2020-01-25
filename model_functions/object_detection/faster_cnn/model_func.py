
from torchvision import transforms
model = None


def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """

    # imports
    import pickle as pk
    from flask import jsonify
    import torchvision.models as models
    from torchvision import transforms
    import urllib.request
    from PIL import Image
    import torch
    import torchvision
    import json
    import requests
    import base64
    from io import BytesIO

    # download & prepare model if necessary
    global model
    if not model:
        # download model from GCS
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
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
    try:
        imfile = request.files['photo']

        filename = "/tmp/to_pred.jpg"
        imfile.save(filename)

    except:
        return "error with file"

    # prepare data

    # For training
    images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
    labels = torch.randint(1, 91, (4, 11))
    images = list(image for image in images)
    targets = []
    for i in range(len(images)):
        d = {}
        d['boxes'] = boxes[i]
        d['labels'] = labels[i]
        targets.append(d)
    output = model(images, targets)
    # For inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)

    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    return (jsonify(img_str), 200, headers)
