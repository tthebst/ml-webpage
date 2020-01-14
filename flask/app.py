from flask import Flask
from flask import render_template
import os
app = Flask(__name__)


@app.route('/')
def home():

    return render_template('home.html')
