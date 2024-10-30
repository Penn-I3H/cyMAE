FROM continuumio/miniconda3:latest

WORKDIR /service

COPY . .

RUN ls /service

RUN mkdir -p data

# Add additional dependencies below ...
RUN conda env create -f environment.yml

ENV PATH=/opt/conda/envs/cymae/bin:$PATH


ENTRYPOINT [ "python3.11", "/service/main.py" ]
# CMD ["python3.11", "/service/main.py"]
