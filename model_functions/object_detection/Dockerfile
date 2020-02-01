FROM python:3.6.10

COPY requirements.txt /


RUN pip install -r /requirements.txt


ADD . /object_detection


EXPOSE 5001

RUN ls /object_detection

CMD [ "python3","/object_detection/app_detect.py" ]