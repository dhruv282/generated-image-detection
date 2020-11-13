#!/usr/bin/python3
import os
import time
import requests
import shutil
import hashlib

NUM_OF_IMAGES_TO_FETCH = 200
URLS = ["https://thispersondoesnotexist.com/image", "https://thiscatdoesnotexist.com"]
DATASET_PATH = "dataset/"

def fetchImages(url, numImages, datasetPath):
	for x in range(numImages):
		r = requests.get(url, stream=True)
		time.sleep(2)
		if r.status_code == 200:
			r.raw.decode_content = True
			filePath = datasetPath+str(x)+'.jpeg'
			with open(filePath, 'wb') as pic:
				shutil.copyfileobj(r.raw, pic)

			print('Image successfully downloaded: '+filePath)
		else:
			print('Unable to retrieve image')

def getHash(string):
	return hashlib.md5(string.encode()).hexdigest()

def main():
	for url in URLS:
		path = DATASET_PATH+str(getHash(url))+'/'

		if not os.path.exists(path):
			os.makedirs(path)

		fetchImages(url, NUM_OF_IMAGES_TO_FETCH, path)

if __name__ == "__main__":
	main()