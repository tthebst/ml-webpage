

deeplabv3_model = None


def deeplabv3(request):
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
    import base64
    from io import BytesIO

    # download & prepare model if necessary
    global deeplabv3_model
    if not deeplabv3_model:
        # download model from GCS
        deeplabv3_model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)
    deeplabv3_model.eval()

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

    input_image = Image.open("/tmp/to_pred.jpg")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    with torch.no_grad():
        output = deeplabv3_model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)

    r = r.convert('RGB')
    buffered = BytesIO()
    r.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    print(type(img_str))
    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    print("done sending response")

    return (jsonify(str(img_str)), 200, headers)
