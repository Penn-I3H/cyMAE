FROM python:3.11.0

WORKDIR /service

RUN apt clean && apt-get update

COPY . .

RUN ls /service

RUN mkdir -p data

RUN pip install -r /service/requirements.txt

ENTRYPOINT [ "python3.11", "/service/main.py" ]