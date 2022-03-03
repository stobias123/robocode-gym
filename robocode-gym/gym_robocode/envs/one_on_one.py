from concurrent import futures
import time
import logging
import subprocess
import gym
from gym import spaces
import logging
import time
import numpy
import json
import http.client
import io, base64
from PIL import Image


class OneOnOneEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self):
        self.__version__ = "0.1.0"
        super(OneOnOneEnv, self).__init__()
        logging.info(f"[OneOnOne Env] - Version {self.__version__}")
        # Env Setup - WASD to integer.
        HEIGHT=600
        WIDTH=800
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(HEIGHT, WIDTH, 3), dtype=numpy.uint8)

        ### Server setup
        self.r = RoboCode()
        self.r.start()
        self.r.remote_client = RemoteClient()
        logging.info(f"[OneOnOne Env] - We're after robocode start")
        self.last_frame = None

    def reset(self):
        self.r.remote_client.reset()
        obs = self.step(0)
        return obs[0]
        #return self.last_frame, obs['reward'], obs['done'], obs['info']


    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        # Send our action
        obs = self.r.remote_client.step(action)
        episode_over = obs['done']
        # self.r.remote_client.writeImage(obs['observation'])
        self.last_frame = self.r.remote_client.obsAsNumpyArray(obs['observation'])
        return self.last_frame, obs['reward'], obs['done'], obs['info']

    def _get_reward(self):
        pass

    def render(self,mode='rgb_array'):
        return self.last_frame

class RoboCode:
    def __init__(self):
        pass

    def start(self):
        logging.info("[RoboCode] Starting Robocode")
        command = [
            "java",
            "-Xmx512M",
            "-DNOSECURITY=true",
            "-DWORKINGDIRECTORY=/Users/stobias/robocode",
            "-cp",
            "libs/*", 
            "-XX:+IgnoreUnrecognizedVMOptions", 
            "--add-opens=java.base/sun.net.www.protocol.jar=ALL-UNNAMED",
            "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
            "--add-opens=java.desktop/javax.swing.text=ALL-UNNAMED",
            "--add-opens=java.desktop/sun.awt=ALL-UNNAMED",
            "robocode.Robocode",
            "-battle",
            "-battle /Users/stobias/repos/misc_projects/robocode/.sandbox/battles/tobybot.battle"
        ]
        #subprocess.run(["/bin/bash","/Users/stobias/robocode/robocode.sh"])
        #subprocess.Popen(command)
        logging.info("[RoboCode] Started Robocode")


class RemoteClient():
    def __init__(self):
        pass

    def reset(self):
        connection = http.client.HTTPConnection('localhost:8000')
        connection.request('GET','/reset')


    def step(self,action: int):
        connection = http.client.HTTPConnection('localhost:8000')
        headers = {'Content-type': 'application/json'}
        actionBlob = {'actionChoice': int(action)}
        jsonAction = json.dumps(actionBlob)
        connection.request('POST','/step',jsonAction,headers)
        resp = connection.getresponse().read().decode()
        return json.loads(resp)

    def writeImage(self, b64String):
        if b64String != b'':
            img = Image.open(io.BytesIO(base64.decodebytes(bytes(b64String, "utf-8"))))
            img.save(f"{int(time.time())}.png")

    def obsAsNumpyArray(self, b64String):
        if b64String == b'' or b64String == '':
            return numpy.empty((600,800,3),dtype=numpy.uint8)
        img = Image.open(io.BytesIO(base64.decodebytes(bytes(b64String, "utf-8"))))
        return numpy.asarray(img, dtype=numpy.uint8)
