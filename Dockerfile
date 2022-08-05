FROM python:3.8


#WORKDIR /model


COPY requirements.txt requirements.txt


RUN set FLASK_APP=flask_backend_with_aws
RUN pip install -r requirements.txt
RUN pip install newsapi-python

EXPOSE 5000

COPY model/ .

CMD ["flask", "run"]