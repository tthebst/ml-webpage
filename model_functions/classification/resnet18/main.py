
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
    import json
    import requests

  # download & prepare model if necessary
    global model
    if not model:
        # download model from GCS
        model = models.resnet18(pretrained=True)
    model.eval()

    # Set CORS headers for preflight requests

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

    # test url
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

    # get imagenet labels
    class_idx = json.loads(urllib.request.urlopen("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json").read())
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    # prepare data
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # create prediction
    with torch.no_grad():
        output = model(input_batch)
    pred = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy().tolist()

    top_pred = []

    top_i = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:5]

    for idx in top_i:
        top_pred.append((idx2label[idx], pred[idx]))

    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    return (jsonify(top_pred), 200, headers)
