FROM python:3.6.10

COPY requirements.txt /


RUN pip install -r /requirements.txt


ADD . /object_detection


EXPOSE 5001

RUN ls /object_detection

RUN pip install gunicorn
WORKDIR /object_detection
CMD exec gunicorn  --bind 0.0.0.0:8080 --workers 1 --threads 2 app_detect:app
#CMD [ "python3","/object_detection/app_detect.py" ]