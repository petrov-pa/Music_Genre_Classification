FROM python:3.7-slim

COPY . /root

WORKDIR /root

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1

RUN pip install -r requirements.txt