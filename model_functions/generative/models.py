

pgan_model = None
# module = None


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


def biggan(request, module, sess, graph, output, inputs, to_pred=42):
    """Responds to any HTTP request.
   Args:
        request(flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response < http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """

    # imports
    import time
    import pickle as pk
    from flask import jsonify
    import urllib.request
    from PIL import Image
    import json
    import requests
    import base64
    from io import BytesIO
    import io
    import numpy as np
    import PIL.Image
    from scipy.stats import truncnorm
    # import tensorflow.compat.v1 as tf
    # To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
    # tf.disable_eager_execution()
    import tensorflow as tf
    import tensorflow_hub as hub
    import matplotlib.pyplot as plt
    # download & prepare model if necessary
    global biggan_model
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    print(tf.__version__)

    # init tensorflow

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

    def imgrid(imarray, cols=5, pad=1):
        if imarray.dtype != np.uint8:
            raise ValueError('imgrid input imarray must be uint8')
        pad = int(pad)
        assert pad >= 0
        cols = int(cols)
        assert cols >= 1
        N, H, W, C = imarray.shape
        rows = N // cols + int(N % cols != 0)
        batch_pad = rows * cols - N
        assert batch_pad >= 0
        post_pad = [batch_pad, pad, pad, 0]
        pad_arg = [[0, p] for p in post_pad]
        imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
        H += pad
        W += pad
        grid = (imarray
                .reshape(rows, cols, H, W, C)
                .transpose(0, 2, 1, 3, 4)
                .reshape(rows*H, cols*W, C))
        if pad:
            grid = grid[:-pad, :-pad]
        return grid

    def imdata(a, format='png', jpeg_fallback=True):
        a = np.asarray(a, dtype=np.uint8)
        data = io.BytesIO()
        PIL.Image.fromarray(a).save(data, format)
        im_data = data.getvalue()
        return im_data

        num_samples = 1  # @param {type:"slider", min:1, max:20, step:1}
        truncation = 0.4  # @param {type:"slider", min:0.02, max:1, step:0.02}
        noise_seed = 0  # @param {type:"slider", min:0, max:100, step:1}

    def one_hot(index, vocab_size):
        index = np.asarray(index)
        if len(index.shape) == 0:
            index = np.asarray([index])
        assert len(index.shape) == 1
        num = index.shape[0]
        output = np.zeros((num, vocab_size), dtype=np.float32)
        output[np.arange(num), index] = 1
        return output

    def one_hot_if_needed(label, vocab_size):
        label = np.asarray(label)
        if len(label.shape) <= 1:
            label = one_hot(label, vocab_size)
        assert len(label.shape) == 2
        return label

    def truncated_z_sample(batch_size, truncation=1., seed=None):
        state = None if seed is None else np.random.RandomState(seed)
        dim_z = inputs['z'].shape.as_list()[1]
        values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
        return truncation * values
    # Load BigGAN 512 module.

    # global module
    pre = time.time()
    """
    if not module:
        tf.reset_default_graph()
        module = hub.Module('https://tfhub.dev/deepmind/biggan-512/2')
    """

    print("loadind module time ", time.time()-pre)
    # Sample random noise (z) and ImageNet label (y) inputs.
    batch_size = 1
    truncation = 0.5  # scalar truncation value in [0.02, 1.0]
    z = truncated_z_sample(batch_size, truncation, 0)
    z = np.asarray(z)  # noise sample

    label = np.asarray(to_pred)
    label = one_hot_if_needed(label, 1000)

    # Call BigGAN on a dict of the inputs to generate a batch of images with shape
    # [8, 512, 512, 3] and range [-1, 1].
    print("creating smaples")
    pre = time.time()

    print("module ", time.time()-pre)
    """
    config = tf.ConfigProto(device_count={'GPU': 0})
    initializer = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(initializer)
    pre = time.time()

    """
    print("running session ", time.time()-pre)
    # with graph.as_default():
    samples = sess.run(output, feed_dict={inputs['z']: z, inputs['y']: label, inputs['truncation']: truncation})
    # samples = sess.run(samples)
    samples = [samples]
    samples = np.concatenate(samples, axis=0)
    # assert samples.shape[0] == num
    samples = np.clip(((samples + 1) / 2.0) * 256, 0, 255)
    samples = np.uint8(samples)
    pre = time.time()
    plt.imsave("/tmp/generated.png", imgrid(samples, cols=min(batch_size, 5)))
    print("creating image ", time.time()-pre)
    with open("/tmp/generated.png", 'rb') as f:
        img_str = base64.b64encode(f.read())
        print(type(img_str))
        print(img_str)
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
    print("done sending response")

    return (jsonify(str(img_str)), 200, headers)
