FROM python:3.7
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app
COPY . /app/
RUN cd /app/robocode-gym && pip install -e .
RUN pip install boto3
