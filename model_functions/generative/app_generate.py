import time
from flask import Flask, request
from flask import render_template
import os
import tensorflow_hub as hub
import tensorflow as tf
import json
import sys
import servingmodels
from tensorflow import keras
import logging
app = Flask(__name__)

module = None
sess = None
graph = None
output = None
inputs = None

# prepares biggan module for faster later use
@app.before_first_request
def gen_module():
    global module
    global sess
    global graph
    global output
    global inputs

    pre = time.time()

    module = hub.Module('https://tfhub.dev/deepmind/biggan-512/2')

    inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
              for k, v in module.get_input_info_dict().items()}
    output = module(inputs)
    graph = tf.get_default_graph()
    print("graph befor first req", graph)


@app.route('/')
def home():

    return "Nothing to see here...."


@app.route('/pgan', methods=["GET"])
def pgan():

    try:
        print("graph calling servingmodel", graph)
        to_send = servingmodels.pgan(request)

    except Exception as e:
        print(e)
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
        print("error")
        return ("ERROR OCCURED", 404, headers)
    return to_send


@app.route('/dcgan', methods=["GET"])
def dcgan():

    try:
        to_send = servingmodels.dcgan(request)

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

    try:
        to_pred = json.loads(request.data.decode())
        to_send = servingmodels.biggan(request, module, output, inputs, graph, to_pred=int(to_pred['a']))
    except Exception as e:

        logger.exception(e)
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
        print("error")
        return ("ERROR OCCURED", 404, headers)
    return to_send


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    app.run(port=5002, debug=True, host="0.0.0.0",  threaded=True)
