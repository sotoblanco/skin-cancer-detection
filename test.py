import requests


#url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

url = "https://oljupdqfdd.execute-api.us-west-2.amazonaws.com/test-skin"

data = {"url": "https://upload.wikimedia.org/wikipedia/commons/6/6c/Melanoma.jpg"}
result = requests.post(url, json=data).json()

print(result)