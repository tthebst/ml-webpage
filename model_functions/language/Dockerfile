FROM tensorflow/tensorflow:1.15.0-py3

COPY requirements.txt /


RUN pip install -r /requirements.txt

RUN pip install torch
RUN pip install torchvision

ADD . /generative


EXPOSE 5002


CMD [ "python3","/generative/app_generate.py" ]