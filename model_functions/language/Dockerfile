FROM ubuntu:18.04



COPY requirements.txt /

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install cython
RUN pip3 install -r /requirements.txt

RUN pip3 install torch
RUN pip3 install torchvision

RUN apt-get install -y libsndfile1

RUN pip3 install soundfile


ADD . /language


EXPOSE 5002
RUN pip3 install gunicorn
WORKDIR /language
CMD exec gunicorn  --bind 0.0.0.0:8080 --workers 1 --threads 2 app_language:app
#CMD [ "python3","/language/app_language.py" ]