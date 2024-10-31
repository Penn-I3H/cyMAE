FROM pytorch/pytorch

WORKDIR /service

RUN apt clean && apt-get update

COPY . .

RUN ls /service

RUN mkdir -p data

# Add additional dependencies below ...
RUN pip install -r /service/requirements.txt
RUN python -V
RUN python3.10 -V

ENTRYPOINT [ "python3.10", "/service/main.py" ]