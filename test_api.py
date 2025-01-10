import requests

# URL of the Flask API
url = 'http://127.0.0.1:5000/predict'

# Path to the image you want to test
image_path = 'Vladimir_putin_0002.jpg'

# Send the image as a POST request
with open(image_path, 'rb') as img_file:
    files = {'image': img_file}
    response = requests.post(url, files=files)

# Print the response
print(response.json())
