FROM tensorflow/tensorflow:2.4.1-gpu

RUN mkdir /workspace
WORKDIR /workspace
COPY app /workspace/app

ENTRYPOINT ["python", "app/entrypoint.py"]