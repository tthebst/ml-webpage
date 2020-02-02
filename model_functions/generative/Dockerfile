FROM python:3.6.10

COPY requirements.txt /


RUN pip install -r /requirements.txt


ADD . /generative


EXPOSE 5002

RUN ls /generative

CMD [ "python3","/generative/app_generate.py" ]