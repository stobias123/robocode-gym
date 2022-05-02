import time
import requests
from datetime import datetime


url = 'http://localhost:8000/step'
body = '{"actionChoice":2}'
headers = {"charset": "utf-8", "Content-Type": "application/json"}


counter = 0
start = datetime.now()
for i in range(1,1000):
  requests.post(url, data=body)
  counter = counter + 1

end = datetime.now()
seconds = (end - start).total_seconds()
cps = counter / seconds
print('--------------')
print(f"called at {cps} / second")
