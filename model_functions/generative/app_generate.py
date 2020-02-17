import time
from flask import Flask, request
from flask import render_template
import os
import tensorflow_hub as hub
import tensorflow as tf
import json
import models
from tensorflow import keras
app = Flask(__name__)

module = None
sess = None
graph = None
output = None
inputs = None


@app.before_first_request
def gen_module():
    global module
    global sess
    global graph
    global output
    global inputs

    pre = time.time()

    tf.reset_default_graph()
    module = hub.Module('https://tfhub.dev/deepmind/biggan-512/2')

    inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
              for k, v in module.get_input_info_dict().items()}
    output = module(inputs)
    """
    config = tf.ConfigProto(device_count={'GPU': 0})
    initializer = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(initializer)
    graph = tf.get_default_graph()
    """


@app.route('/')
def home():

    return "Nothing to see here...."


@app.route('/pgan', methods=["GET"])
def pgan():

    try:
        to_send = models.pgan()

    except Exception as e:
        print(e)
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
        print("error")
        return ("ERROR OCCURED", 404, headers)
    return to_send


@app.route('/biggan', methods=["GET", "POST"])
def biggan():
    to_pred = json.loads(request.data.decode())
    print("requesting gan", sess, graph)
    to_send = models.biggan(request, module, output, inputs, to_pred=int(to_pred['a']))
    try:
        pass
    except Exception as e:
        print(e)
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
        print("error")
        return ("ERROR OCCURED", 404, headers)
    return to_send


if __name__ == '__main__':
    app.run(port=5002, debug=True, host="0.0.0.0",  threaded=False)
