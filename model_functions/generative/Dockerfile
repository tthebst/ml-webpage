FROM tensorflow/tensorflow:1.15.2-py3

COPY requirements.txt /


RUN pip install -r /requirements.txt

ADD . /generative


EXPOSE 5002

RUN ls /generative
RUN pip list


CMD [ "python3","/generative/app_generate.py" ]