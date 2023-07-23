FROM python:3.6.10

ENV PYTHONUNBUFFERED 1

WORKDIR /opt
ADD requirements.txt /opt/requirements.txt
RUN pip install -r requirements.txt

COPY . /opt
ENTRYPOINT ["./entrypoint.sh"]
EXPOSE 8000  