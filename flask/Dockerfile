FROM ubuntu:18.04


COPY requirements.txt /


RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install cython
RUN apt-get install -y libsndfile1
RUN pip3 install -r /requirements.txt

ADD . /app

RUN ls



WORKDIR /app
RUN ls
CMD exec gunicorn  --bind 0.0.0.0:5000 --workers 1 --threads 2 app:app