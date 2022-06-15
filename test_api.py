import requests

url = "https://fuel-receipt-detection.herokuapp.com/api/v1/scan"
img_path = "path/to/images/image.jpg"

username = None  # add username here
password = None  # add password here

response = requests.post(
    url, files={"file": (img_path, open(img_path, "rb"))}, auth=(username, password)
)
print(response.text)
