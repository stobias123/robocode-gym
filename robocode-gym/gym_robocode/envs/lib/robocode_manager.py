import logging
import docker
import time


class RobocodeManager:

    def __init__(self, port_number: int, robocode_image: str = 'gcr.io/stobias-dev/robocode:12a6d0da5'):
        self.robocode_image: str = robocode_image
        self.port_number: int = None

    def start(self):
        pass

    def stop(self):
        pass


class RobocodeManagerImpl(RobocodeManager):
    def __init__(self, port_number):
        self.docker_client = docker.from_env()
        self.port_number = port_number
        self.robocode_image = 'gcr.io/stobias-dev/robocode:12a6d0da5'
        self.container = None

    # docker run -it --net=host -d --name robocode stobias123/robocode
    def start(self):
        logging.info(f"[RoboCode] Starting Robocode on port {self.port_number}")
        self.container = self.docker_client.containers.run(self.robocode_image,
                                                           detach=True,
                                                           auto_remove=True,
                                                           ports={
                                                               8000: self.port_number
                                                           })
        logging.info(f"Started container {self.container.id}")
        time.sleep(5)
        logging.info(f"[RoboCode] Started Robocode on port {self.port_number}")

    def stop(self):
        self.container.stop()
