import time
import numpy
import json
import logging
import http.client
import io, base64
from PIL import Image

class ConnectionManager():
    def __init__(self,port_number,hostname='localhost'):
        self.connection_url = f"{hostname}:{port_number}"
        print(f"Connection manager started with connection url - {self.connection_url}")

    def reset(self):
        logging.info('[ConnectionManager] Resetting')
        connection = http.client.HTTPConnection(self.connection_url)
        connection.request('GET','/reset')

    def step(self,action: int):
        connection = http.client.HTTPConnection(self.connection_url)
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
