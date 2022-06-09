import requests

base_url = 'http://127.0.0.1:5000'

response = requests.get(f"{base_url}")
print(response.json())
