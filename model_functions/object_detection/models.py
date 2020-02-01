

pgan_model = None
fastrnn_model = None
fastrnn_model2 = None


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


def fastrnn(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """

    # imports
    try:
        import pickle as pk
        from flask import jsonify
        import urllib.request
        from PIL import Image
        import json
        import requests
        from PIL import Image
        from PIL import ImageColor
        from PIL import ImageDraw
        from PIL import ImageFont
        import numpy as np
        from PIL import ImageOps
        import tensorflow as tf
        import tensorflow_hub as hub
        import matplotlib.pyplot as plt
        import tempfile
        from six.moves.urllib.request import urlopen
        from six import BytesIO
        import base64
        import time

        # helper functions

        def display_image(image):
            fig = plt.figure(figsize=(20, 15))
            plt.grid(False)

            plt.imsave("/tmp/prediction.png", image, format="png")

        def download_and_resize_image(url, new_width=256, new_height=256, display=False):
            _, filename = tempfile.mkstemp(suffix=".jpg")
            response = urlopen(url)
            image_data = response.read()
            image_data = BytesIO(image_data)
            pil_image = Image.open(image_data)
            pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
            pil_image_rgb = pil_image.convert("RGB")
            pil_image_rgb.save(filename, format="JPEG", quality=90)
            print("Image downloaded to %s." % filename)
            if display:
                display_image(pil_image)
            return filename

        def draw_bounding_box_on_image(image,
                                       ymin,
                                       xmin,
                                       ymax,
                                       xmax,
                                       color,
                                       font,
                                       thickness=4,
                                       display_str_list=()):
            """Adds a bounding box to an image."""
            draw = ImageDraw.Draw(image)
            im_width, im_height = image.size
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                       (left, top)],
                      width=thickness,
                      fill=color)

            # If the total height of the display strings added to the top of the bounding
            # box exceeds the top of the image, stack the strings below the bounding box
            # instead of above.
            display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
            # Each display_str has a top and bottom margin of 0.05x.
            total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

            if top > total_display_str_height:
                text_bottom = top
            else:
                text_bottom = bottom + total_display_str_height
            # Reverse list and print from bottom to top.
            for display_str in display_str_list[::-1]:
                text_width, text_height = font.getsize(display_str)
                margin = np.ceil(0.05 * text_height)
                draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                                (left + text_width, text_bottom)],
                               fill=color)
                draw.text((left + margin, text_bottom - text_height - margin),
                          display_str,
                          fill="black",
                          font=font)
                text_bottom -= text_height - 2 * margin

        def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
            """Overlay labeled boxes on an image with formatted scores and label names."""
            colors = list(ImageColor.colormap.values())

            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                          25)
            except IOError:
                print("Font not found, using default font.")
                font = ImageFont.load_default()

            for i in range(min(boxes.shape[0], max_boxes)):
                if scores[i] >= min_score:
                    ymin, xmin, ymax, xmax = tuple(boxes[i])
                    display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                                   int(100 * scores[i]))
                    color = colors[hash(class_names[i]) % len(colors)]
                    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
                    draw_bounding_box_on_image(
                        image_pil,
                        ymin,
                        xmin,
                        ymax,
                        xmax,
                        color,
                        font,
                        display_str_list=[display_str])
                    np.copyto(image, np.array(image_pil))
            return image

        def load_img(path):
            img = tf.io.read_file(path)

            img = tf.image.decode_jpeg(img, channels=3)
            return img

        def run_detector(detector, path):
            img = load_img(path)
            print(path)
            converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
            start_time = time.time()
            result = detector(converted_img)
            end_time = time.time()

            result = {key: value.numpy() for key, value in result.items()}

            print("Found %d objects." % len(result["detection_scores"]))
            print("Inference time: ", end_time-start_time)

            image_with_boxes = draw_boxes(
                img.numpy(), result["detection_boxes"],
                result["detection_class_entities"], result["detection_scores"])

            display_image(image_with_boxes)

    # download & prepare model if necessary
        global fastrnn_model
        print(tf.__version__)
        if not fastrnn_model:
            # download model from GCS
            # @param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
            module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"

            fastrnn_model = hub.load(module_handle).signatures['default']

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

        # making predictions
        run_detector(fastrnn_model, "/tmp/to_pred.jpg")

        # make file to img string
        with open("/tmp/prediction.png", "rb") as image:
            b64string = base64.b64encode(image.read())

        headers = {
            'Access-Control-Allow-Origin': '*'
        }

        return (jsonify(str(b64string)), 200, headers)

    except Exception as e:
        print(e)
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
        return ("error has occured", 404, headers)


def fastrnn2(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """

    # imports
    try:
        import pickle as pk
        from flask import jsonify
        import urllib.request
        from PIL import Image
        import json
        import requests
        from PIL import Image
        from PIL import ImageColor
        from PIL import ImageDraw
        from PIL import ImageFont
        import numpy as np
        from PIL import ImageOps
        import tensorflow as tf
        import tensorflow_hub as hub
        import matplotlib.pyplot as plt
        import tempfile
        from six.moves.urllib.request import urlopen
        from six import BytesIO
        import base64
        import time

        # helper functions

        def display_image(image):
            fig = plt.figure(figsize=(20, 15))
            plt.grid(False)

            plt.imsave("/tmp/prediction.png", image, format="png")

        def download_and_resize_image(url, new_width=256, new_height=256, display=False):
            _, filename = tempfile.mkstemp(suffix=".jpg")
            response = urlopen(url)
            image_data = response.read()
            image_data = BytesIO(image_data)
            pil_image = Image.open(image_data)
            pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
            pil_image_rgb = pil_image.convert("RGB")
            pil_image_rgb.save(filename, format="JPEG", quality=90)
            print("Image downloaded to %s." % filename)
            if display:
                display_image(pil_image)
            return filename

        def draw_bounding_box_on_image(image,
                                       ymin,
                                       xmin,
                                       ymax,
                                       xmax,
                                       color,
                                       font,
                                       thickness=4,
                                       display_str_list=()):
            """Adds a bounding box to an image."""
            draw = ImageDraw.Draw(image)
            im_width, im_height = image.size
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                       (left, top)],
                      width=thickness,
                      fill=color)

            # If the total height of the display strings added to the top of the bounding
            # box exceeds the top of the image, stack the strings below the bounding box
            # instead of above.
            display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
            # Each display_str has a top and bottom margin of 0.05x.
            total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

            if top > total_display_str_height:
                text_bottom = top
            else:
                text_bottom = bottom + total_display_str_height
            # Reverse list and print from bottom to top.
            for display_str in display_str_list[::-1]:
                text_width, text_height = font.getsize(display_str)
                margin = np.ceil(0.05 * text_height)
                draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                                (left + text_width, text_bottom)],
                               fill=color)
                draw.text((left + margin, text_bottom - text_height - margin),
                          display_str,
                          fill="black",
                          font=font)
                text_bottom -= text_height - 2 * margin

        def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
            """Overlay labeled boxes on an image with formatted scores and label names."""
            colors = list(ImageColor.colormap.values())

            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                          25)
            except IOError:
                print("Font not found, using default font.")
                font = ImageFont.load_default()

            for i in range(min(boxes.shape[0], max_boxes)):
                if scores[i] >= min_score:
                    ymin, xmin, ymax, xmax = tuple(boxes[i])
                    display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                                   int(100 * scores[i]))
                    color = colors[hash(class_names[i]) % len(colors)]
                    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
                    draw_bounding_box_on_image(
                        image_pil,
                        ymin,
                        xmin,
                        ymax,
                        xmax,
                        color,
                        font,
                        display_str_list=[display_str])
                    np.copyto(image, np.array(image_pil))
            return image

        def load_img(path):
            img = tf.io.read_file(path)
            print(img.get_shape())
            img = tf.image.decode_jpeg(img, channels=3, ratio=4)
            return img

        def run_detector(detector, path):
            img = load_img(path)
            print(path)
            converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
            start_time = time.time()
            result = detector(converted_img)
            end_time = time.time()

            result = {key: value.numpy() for key, value in result.items()}

            print("Found %d objects." % len(result["detection_scores"]))
            print("Inference time: ", end_time-start_time)

            image_with_boxes = draw_boxes(
                img.numpy(), result["detection_boxes"],
                result["detection_class_entities"], result["detection_scores"])

            display_image(image_with_boxes)

    # download & prepare model if necessary
        global fastrnn_model2
        print(tf.__version__)
        if not fastrnn_model2:
            # download model from GCS
            # @param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
            module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

            fastrnn_model2 = hub.load(module_handle).signatures['default']

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

        # making predictions
        run_detector(fastrnn_model2, "/tmp/to_pred.jpg")

        # make file to img string
        with open("/tmp/prediction.png", "rb") as image:
            b64string = base64.b64encode(image.read())

        headers = {
            'Access-Control-Allow-Origin': '*'
        }

        return (jsonify(str(b64string)), 200, headers)

    except Exception as e:
        print(e)
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
        return ("error has occured", 404, headers)
