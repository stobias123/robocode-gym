FROM python:3.7
COPY requirements.txt /app/
RUN pip3 install -r /app/requirements.txt
COPY . /app/

WORKDIR /app
RUN cd /app/robocode-gym && pip install -e .
RUN git clone https://github.com/stobias123/rl-baselines3-zoo && echo foo
WORKDIR /app/rl-baselines3-zoo
RUN pip3 install -r requirements.txt
COPY run.sh /
CMD ["/run.sh"]
