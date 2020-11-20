#!/usr/bin/python3
import sys
import os
import subprocess

def convertVideoToImages(videoPath, imagesDir):
	print("...converting video to images")
	print("Video: "+videoPath)
	
	if imagesDir[-1] != '/':
		imagesDir += '/'
	imageNames = imagesDir+videoPath.split('/')[-1].replace('.mp4', '') + '-%d.jpg'
	
	conv = subprocess.run(['ffmpeg', '-i', videoPath, imageNames, '-hide_banner', '-loglevel', 'panic'])

	if conv.returncode == 0:
		print("...conversion successful!")
		print("Images stored in: " + imagesDir)
	else:
		print("Error: Something went wrong :(")


def processAllVideos(datasetPath):
	if datasetPath[-1] != '/':
		datasetPath += '/'
	for sequences in os.listdir(datasetPath):
		if 'sequences' in sequences:
			for faceTools in os.listdir(datasetPath+sequences):
				for video in os.listdir(datasetPath+sequences+'/'+faceTools+'/c23/videos/'):
					videoPath = datasetPath+sequences+'/'+faceTools+'/c23/videos/'+video
					
					imagesDir = videoPath.replace('videos', 'images')
					imagesDir = imagesDir.replace('.mp4', '/')

					if not os.path.exists(imagesDir):
						os.makedirs(imagesDir)
						convertVideoToImages(videoPath, imagesDir)



def main():
	if len(sys.argv) != 2:
		print("Usage: "+ sys.argv[0] + " <dataset path>")
		sys.exit()
	processAllVideos(sys.argv[1])

if __name__ == "__main__":
	main()