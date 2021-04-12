FROM python:3.9.4-buster

RUN apt-get update && \
    apt-get install -y openslide-tools libpq-dev libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /workspace
WORKDIR /workspace

COPY requirements.txt /workspace
RUN pip install numpy==1.20.2
RUN pip install -r requirements.txt

COPY app /workspace/app

ENTRYPOINT ["python", "app/entrypoint.py"]