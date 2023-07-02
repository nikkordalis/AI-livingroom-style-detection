FROM python:3.8-buster

COPY styleup styleup
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY Makefile Makefile

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn app.simple:app --host 0.0.0.0
