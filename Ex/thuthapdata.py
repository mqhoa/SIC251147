import requests as rq

res = rq.get('https://en.wikipedia.org/wiki')
res.status_code
print(res.text)
