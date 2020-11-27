#!/usr/bin/python3
import os
import sys
import time
import requests
import shutil
import hashlib

#NUM_OF_IMAGES_TO_FETCH = 200
URLS = ["https://thispersondoesnotexist.com/image"]
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

	if len(sys.argv) == 1:
		NUM_OF_IMAGES_TO_FETCH = 200
	elif (len(sys.argv) == 2):
		NUM_OF_IMAGES_TO_FETCH = int(sys.argv[1])
	else:
		print("\nUSAGE")
		print("The program takes a single argument, the number of images to download.")
		print("If no argument is passed, the program defaults to 200.\n")
		print("Example:")
		print("python3 fetchGeneratedImages.py 500\n")
		print("or to default to 200\n")
		print("python3 fetchGeneratedImages.py\n")
		sys.exit()

	for url in URLS:
		path = DATASET_PATH+str(getHash(url))+'/'

		if not os.path.exists(path):
			os.makedirs(path)

		fetchImages(url, NUM_OF_IMAGES_TO_FETCH, path)

if __name__ == "__main__":
	main()