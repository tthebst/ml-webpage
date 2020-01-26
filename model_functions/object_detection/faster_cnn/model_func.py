
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

    # define box drawing function
    def bbox_to_rect(bbox, color):
        return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1], fill=False, edgecolor=color, linewidth=2)

    # transform input image
    input_image = Image.open(filename)
    tarans = torchvision.transforms.ToTensor()
    input_image = tarans(input_image)
    model.eval()

    predictions = model([input_image])

    # draw image with boxes
    image = [np.moveaxis(image[0].cpu().detach().numpy(), 0, -1)]
    fig = plt.imshow(image[0])
    for boxes in predictions[0]['boxes'].data.cpu().detach().numpy().tolist():
        fig.axes.add_patch(bbox_to_rect(boxes, 'blue'))

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue())
    buf.close()

    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    return (jsonify(image_base64), 200, headers)
