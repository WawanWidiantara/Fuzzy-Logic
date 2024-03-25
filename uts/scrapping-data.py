import requests

url = "https://iprice.co.id/harga/ms-glow-facial-wash/"
response = requests.get(url)
response.raise_for_status()  # Check for successful request

html_content = response.text
