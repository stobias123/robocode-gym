FROM python:3.7
COPY requirements.txt /app/
RUN pip3 install -r /app/requirements.txt

WORKDIR /app
COPY . /app/
RUN cd /app/robocode-gym && pip install -e .