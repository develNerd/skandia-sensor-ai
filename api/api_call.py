import requests

image_path = "../IMG_4236.JPG"
url = "http://localhost:5000/predict"
files = {"image": open(image_path, "rb")}

response = requests.post(url, files=files)
print(response.json())
