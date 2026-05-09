import os
import requests
import zipfile

url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
save_path = "tiny-imagenet-200.zip"
extract_path = "./data"

# 다운로드
print("Downloading Tiny ImageNet...")
response = requests.get(url, stream=True)
with open(save_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

# 압축 해제
print("Extracting files...")
with zipfile.ZipFile(save_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Done!")