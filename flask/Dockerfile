FROM ubuntu:18.04


COPY requirements.txt /


RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install cython
RUN apt-get install -y libsndfile1
RUN pip3 install -r /requirements.txt

ADD . /app





WORKDIR /app
CMD exec gunicorn --ca-certs=/my_certs/etc/letsencrypt/live/ai-demo.ch/fullchain.pem --certfile=/my_certs/etc/letsencrypt/live/ai-demo.ch/cert.pem --keyfile=/my_certs/etc/letsencrypt/live/ai-demo.ch/privkey.pem --ssl-version=3 --bind 0.0.0.0:443 --workers 1 --threads 2 app:app
#CMD exec gunicorn  --bind 0.0.0.0:443 --workers 1 --threads 2 app:app