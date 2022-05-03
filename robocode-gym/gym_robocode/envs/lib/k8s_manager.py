import os
from random import randint
from gym_robocode.envs.lib.robocode_manager import *
import time
from kubernetes import client, config, watch


class K8sManager(RobocodeManager):
    def __init__(self,  namespace: str, port_number=8000, robocode_image: str = 'gcr.io/stobias-dev/robocode'):
        super().__init__(port_number=port_number, robocode_image=robocode_image)
        print("loading in cluster config")
        config.load_incluster_config()
        self.v1_client = client.CoreV1Api()
        self.namespace = namespace
        self.port_number = port_number
        self.ip = None
        self.pod_name = f"robocode-training-{randint(1,10000)}"
        self.pod_labels = {"job_name": self.pod_name}

    # docker run -it --net=host -d --name robocode stobias123/robocode
    def start(self):
        logging.info(f"[RoboCode] Starting Robocode on port {self.port_number}")
        self.ip = self.create_robocode_game()
        time.sleep(5)
        logging.info(f"[RoboCode] Started Robocode on port {self.port_number}")

    def create_robocode_game(self):
        metadata = client.V1ObjectMeta(generate_name=self.pod_name, labels=self.pod_labels)
        container = client.V1Container(
            image=self.robocode_image,
            name="train",
            image_pull_policy='IfNotPresent',
            ports=[client.V1ContainerPort(
                container_port=self.port_number
            )]
            # args=[],
            # command=["python3", "-u", "./shuffler.py"],
        )
        pod = client.V1Pod(
            spec=client.V1PodSpec(
                restart_policy="Never",
                containers=[container]
            ),
            metadata=client.V1ObjectMeta(name=self.pod_name, labels=self.pod_labels),
        )
        pod = self.v1_client.create_namespaced_pod(namespace=self.namespace, body=pod)
        pod_list = self.v1_client.list_namespaced_pod(label_selector=f"job_name={self.pod_name}",
                                                          namespace=self.namespace)
        pod = pod_list.items[0]
        while pod.status.pod_ip == None:
            logging.info(f"[RoboCode] Checking for pod.")
            pod = self.v1_client.list_namespaced_pod(label_selector=f"job_name={self.pod_name}",
                                                          namespace=self.namespace).items[0]
            time.sleep(1)
        logging.info(f"[RoboCode] Started Robocode at IP {pod.status.pod_ip}")
        return pod.status.pod_ip

    def get_robocode_pod(self):
        pod_list = self.v1_client.list_namespaced_pod(label_selector=f"job_name={self.pod_name}",
                                                          namespace=self.namespace)
        if len(pod_list.items) < 1:
            return None
        if len(pod_list.items) > 1:
            raise "There's a problem, we got too many pods"
        return pod_list.items[0]

    def stop(self):
        self.v1_client.delete_namespaced_pod(label_selector=f"job_name={self.pod_name}",
                                                          namespace=self.namespace)
