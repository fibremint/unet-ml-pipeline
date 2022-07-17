FROM python:3.9-alpine

RUN mkdir /workspace
WORKDIR /workspace
COPY app /workspace/app

ENTRYPOINT ["python", "app/entrypoint.py"]