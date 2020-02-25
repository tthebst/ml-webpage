FROM tensorflow/tensorflow:1.15.0-py3

COPY requirements.txt /


RUN pip install -r /requirements.txt

RUN pip install torch
RUN pip install torchvision

ADD . /generative

RUN pip install gunicorn memory_profiler


WORKDIR /generative
CMD exec gunicorn  --bind 0.0.0.0:5002 --workers 1 --threads 2 app_generate:app
#CMD [ "python3","/generative/app_generate.py" ]