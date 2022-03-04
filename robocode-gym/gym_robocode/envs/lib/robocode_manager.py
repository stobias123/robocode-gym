from concurrent import futures
import logging
import logging
from PIL import Image
import docker
import time

class RoboCodeManager:
    def __init__(self,port_number):
        self.docker_client = docker.from_env()
        self.port_number = port_number
        self.robocode_image = 'stobias123/robocode'
        self.container = None

# docker run -it --net=host -d --name robocode stobias123/robocode
    def start(self):
        logging.info(f"[RoboCode] Starting Robocode on port {self.port_number}")
        container = self.docker_client.containers.run(self.robocode_image,
                                      detach=True,
                                      ports= {
                                          8000: self.port_number 
                                      })

        time.sleep(15)
        logging.info(f"[RoboCode] Started Robocode on port {self.port_number}")

    def stop(self):
        self.container.remove(force=True)