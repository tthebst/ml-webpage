

pgan_model = None
biggan_model = None


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


def biggan(request):
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
    import json
    import requests
    import base64
    from io import BytesIO
    import io
    import numpy as np
    import PIL.Image
    from scipy.stats import truncnorm
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

    print("sess1")
    global graph
    tf_config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    graph = tf.get_default_graph()
    initializer = tf.global_variables_initializer()
    sess = tf.Session(graph=graph, config=tf_config)
    with graph.as_default():
        sess.run(initializer)
        # tf.reset_default_graph()
        if not biggan_model:
            # download model from GCS
            biggan_path = 'https://tfhub.dev/deepmind/biggan-deep-128/1'

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

        print('Loading BigGAN module from:', biggan_path)
        biggan_model = hub.Module(biggan_path)
        inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
                  for k, v in biggan_model.get_input_info_dict().items()}
        output = biggan_model(inputs)

        print('Inputs:\n', '\n'.join(
            '  {}: {}'.format(*kv) for kv in inputs.items()))
        print()
        print('Output:', output)
        input_z = inputs['z']
        input_y = inputs['y']
        input_trunc = inputs['truncation']

        dim_z = input_z.shape.as_list()[1]
        vocab_size = input_y.shape.as_list()[1]

        def truncated_z_sample(batch_size, truncation=1., seed=None):
            state = None if seed is None else np.random.RandomState(seed)
            values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
            return truncation * values

        def one_hot(index, vocab_size=vocab_size):
            index = np.asarray(index)
            if len(index.shape) == 0:
                index = np.asarray([index])
            assert len(index.shape) == 1
            num = index.shape[0]
            output = np.zeros((num, vocab_size), dtype=np.float32)
            output[np.arange(num), index] = 1
            return output

        def one_hot_if_needed(label, vocab_size=vocab_size):
            label = np.asarray(label)
            if len(label.shape) <= 1:
                label = one_hot(label, vocab_size)
            assert len(label.shape) == 2
            return label

        def sample(sess, noise, label, output, truncation=1., batch_size=8, vocab_size=vocab_size):
            noise = np.asarray(noise)
            label = np.asarray(label)
            num = noise.shape[0]
            if len(label.shape) == 0:
                label = np.asarray([label] * num)
            if label.shape[0] != num:
                raise ValueError('Got # noise samples ({}) != # label samples ({})'
                                 .format(noise.shape[0], label.shape[0]))
            label = one_hot_if_needed(label, vocab_size)
            ims = []
            print(num)
            for batch_start in range(0, num, batch_size):
                s = slice(batch_start, min(num, batch_start + batch_size))
                feed_dict = {input_z: noise[s], input_y: label[s], input_trunc: truncation}
                print("LOOOS")
                ims.append(sess.run(output, feed_dict=feed_dict))
                print("ddd")
            ims = np.concatenate(ims, axis=0)
            assert ims.shape[0] == num
            ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
            ims = np.uint8(ims)
            return ims

        def interpolate(A, B, num_interps):
            if A.shape != B.shape:
                raise ValueError('A and B must have the same shape to interpolate.')
            alphas = np.linspace(0, 1, num_interps)
            return np.array([(1-a)*A + a*B for a in alphas])

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

        print("predicting")
        z = truncated_z_sample(num_samples, truncation, noise_seed)
        y = 933
        print("predicting")
        ims = sample(sess, z, y, truncation=truncation, output=output)
        print("done")
        plt.imsave("/tmp/generated.jpeg", imgrid(ims, cols=min(num_samples, 5)))

    with open("/tmp/generated.jpeg", rb) as f:
        img_str = base64.b64encode(f.read())
        print(type(img_str))
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
    print("done sending response")

    return (jsonify(str(img_str)), 200, headers)
