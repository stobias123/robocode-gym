from random import randint
from gym_robocode.envs.lib.robocode_manager import *
import time
from kubernetes import client, config, watch


class K8sManager(RobocodeManager):
    def __init__(self, port_number, namespace: str, robocode_image: str = 'gcr.io/stobias-dev/robocode'):
        super().__init__(port_number=port_number, robocode_image=robocode_image)
        config.load_kube_config()
        self.v1_client = client.CoreV1Api()
        self.namespace = namespace
        self.port_number = port_number
        self.container = None

    # docker run -it --net=host -d --name robocode stobias123/robocode
    def start(self):
        logging.info(f"[RoboCode] Starting Robocode on port {self.port_number}")
        self.create_robocode_game()
        time.sleep(5)
        logging.info(f"[RoboCode] Started Robocode on port {self.port_number}")

    def create_robocode_game(self):
        pod_name = f"robocode-training-{randint(1,10000)}"
        metadata = client.V1ObjectMeta(generate_name=pod_name, labels={"job_name": pod_name})
        container = client.V1Container(
            image=self.robocode_image,
            name="train",
            image_pull_policy='IfNotPresent'
            # args=[],
            # command=["python3", "-u", "./shuffler.py"],
        )
        pod = client.V1Pod(
            spec=client.V1PodSpec(
                restart_policy="Never",
                containers=[container]
            ),
            metadata=client.V1ObjectMeta(name=pod_name),
        )
        pod = self.v1_client.create_namespaced_pod(namespace=self.namespace, body=pod)
        api_response = self.v1_client.list_namespaced_pod(label_selector=f"job_name={pod_name}",
                                                          namespace=self.namespace)
        logging.info(f"[RoboCode] Started Robocode on port {api_response}")

        return pod

    def stop(self):
        self.container.stop()
