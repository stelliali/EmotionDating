import requests


results = {
    "Jonas": 10,
    "Max": 20,
    "Philip": 30
}
r = requests.post("http://localhost:5000/upload_results", json=results)
print(r)