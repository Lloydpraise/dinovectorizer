import requests

# A tiny 1x1 pixel image in base64
dummy_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

# Replace with your actual Leapcell URL
url = "https://dinovectorizer-lloydpraise2265-dlcy9mdo.leapcell.dev"  

payload = {
    "image": f"data:image/png;base64,{dummy_image_b64}"
}

print("Sending image to API...")
response = requests.post(url, json=payload)

print(f"Status Code: {response.status_code}")
try:
    print("Vector Output:", response.json())
except:
    print("Raw text:", response.text)