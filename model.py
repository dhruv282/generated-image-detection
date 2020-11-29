#!/usr/bin/python3
import sys
import os
import shutil
from torchvision import transforms
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch
from PIL import Image
import cv2
import dlib
from videoToImages import convertVideoToImages
from random import shuffle


class SeparableConv2d(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
		super(SeparableConv2d,self).__init__()

		self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
		self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

	def forward(self,x):
		x = self.conv1(x)
		x = self.pointwise(x)
		return x


class Block(nn.Module):
	def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
		super(Block, self).__init__()

		if out_filters != in_filters or strides!=1:
			self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
			self.skipbn = nn.BatchNorm2d(out_filters)
		else:
			self.skip=None

		self.relu = nn.ReLU(inplace=True)
		rep=[]

		filters=in_filters
		if grow_first:
			rep.append(self.relu)
			rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
			rep.append(nn.BatchNorm2d(out_filters))
			filters = out_filters

		for i in range(reps-1):
			rep.append(self.relu)
			rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
			rep.append(nn.BatchNorm2d(filters))

		if not grow_first:
			rep.append(self.relu)
			rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
			rep.append(nn.BatchNorm2d(out_filters))

		if not start_with_relu:
			rep = rep[1:]
		else:
			rep[0] = nn.ReLU(inplace=False)

		if strides != 1:
			rep.append(nn.MaxPool2d(3,strides,1))
		self.rep = nn.Sequential(*rep)

	def forward(self,inp):
		x = self.rep(inp)

		if self.skip is not None:
			skip = self.skip(inp)
			skip = self.skipbn(skip)
		else:
			skip = inp

		x+=skip
		return x


class Xception(nn.Module):
	def __init__(self, num_classes=2):
		super(Xception, self).__init__()
		self.num_classes = num_classes

		self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
		self.bn1 = nn.BatchNorm2d(32)
		self.relu = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv2d(32,64,3,bias=False)
		self.bn2 = nn.BatchNorm2d(64)

		self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
		self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
		self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

		self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
		self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
		self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
		self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

		self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
		self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
		self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
		self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

		self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

		self.conv3 = SeparableConv2d(1024,1536,3,1,1)
		self.bn3 = nn.BatchNorm2d(1536)

		self.conv4 = SeparableConv2d(1536,2048,3,1,1)
		self.bn4 = nn.BatchNorm2d(2048)

		self.last_linear = nn.Linear(2048, num_classes)



	def features(self, input):
		x = self.conv1(input)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)

		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.block5(x)
		x = self.block6(x)
		x = self.block7(x)
		x = self.block8(x)
		x = self.block9(x)
		x = self.block10(x)
		x = self.block11(x)
		x = self.block12(x)

		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu(x)

		x = self.conv4(x)
		x = self.bn4(x)
		return x

	def logits(self, features):
		x = self.relu(features)

		x = F.adaptive_avg_pool2d(x, (1, 1))
		x = x.view(x.size(0), -1)
		x = self.last_linear(x)
		return x

	def forward(self, input):
		x = self.features(input)
		x = self.logits(x)
		return x

def getFaceCrop(image):
	detector = dlib.get_frontal_face_detector()
	faces = detector(image, 1)
	if len(faces) > 0:
		face = faces[0]
		height, width = image.shape[:2]

		scale = 1.3
		x1 = face.left()
		y1 = face.top()
		x2 = face.right()
		y2 = face.bottom()

		size = int(max(x2 - x1, y2 - y1) * scale)
		center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

		x = max(int(center_x - size // 2), 0)
		y = max(int(center_y - size // 2), 0)

		size = min(width - x, size)
		size = min(height - y, size)

		croppedFace = image[y:y+size, x:x+size]

		return croppedFace


def processImages(imagePaths):
	#print("...processing image")
	#print("Image: "+imagePath)
	
	process = transforms.Compose([
			transforms.Resize((299, 299)),
			transforms.ToTensor(),
			transforms.Normalize([0.5]*3, [0.5]*3)
		])


	images = Variable(torch.randn(len(imagePaths), 3, 299, 299).type(torch.FloatTensor), requires_grad=False)
	for i,path in enumerate(imagePaths):
		img = cv2.imread(path)
		#img = getFaceCrop(img)
		img = Image.fromarray(img)
	
		processedImg = process(img)
		processedImg = processedImg.unsqueeze(0)
		if torch.cuda.is_available():
			processedImg = processedImg.cuda()

		images[i] = processedImg
	
	if torch.cuda.is_available():
		images = images.cuda()
	
	return images


def createBatch(dataList, batchSize):
	for i in range(0, len(dataList), batchSize):
		yield dataList[i:i+batchSize]

def trainModel(model, batch, targets, optimizer, loss):
	img = processImages(batch)
	output = model(img)

	prob,_ = torch.max(output, 1)

	targets = torch.LongTensor(targets)
	if torch.cuda.is_available():
		targets = targets.cuda()

	lossVal = loss(output,targets)
	lossVal.backward()
	optimizer.step()

	print('Loss: '+str(lossVal.item()))
	print('\n')

def trainFullNetwork(model, filePaths, tags, batchSize, lr, epochs, data):

	# create batch for GAN images
	if data == 'GAN':
		filePaths = list(createBatch(filePaths, batchSize))
		tags = list(createBatch(tags, batchSize))

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	loss = torch.nn.CrossEntropyLoss()
	if torch.cuda.is_available():
		loss = loss.cuda()

	for epoch in range(epochs):
		print('Epoch: '+str(epoch))
		if data == 'GAN':
			for i in range(len(filePaths)):
				batch = filePaths[i]
				tagBatch = tags[i]

				trainModel(model, batch, tagBatch, optimizer, loss)
		elif data == 'faceForensics':
			for i in range(len(filePaths)):
				path = filePaths[i]
				target = tags[i]

				imagesDir = 'tempTrainingImages/'

				if not os.path.exists(imagesDir):
					os.makedirs(imagesDir)
				convertVideoToImages(path, imagesDir)

				batches = []
				targets = []
				for imagePath in os.listdir(imagesDir):
					imagePath = imagesDir + imagePath
					batches.append(imagePath)
					targets.append(target)
				
				batches = list(createBatch(batches, batchSize))
				targets = list(createBatch(targets, batchSize))
				
				for i in range(len(batches)):
					batch = batches[i]
					tagBatch = targets[i]

					trainModel(model, batch, tagBatch, optimizer, loss)

				# delete images
				shutil.rmtree(imagesDir)

		saveModel(model, data)


def saveModel(model, data):
	# save model
	modelPath = ''
	if data == 'faceForensics':
		modelPath = 'faceForensics'
	elif data == 'GAN':
		modelPath += 'GAN'
	modelPath += '_model.pth'

	torch.save(model.state_dict(), modelPath)
	print('Model successfully saved!\n\n')


def predict(model, imagePath, postFunc=nn.Softmax(dim=1)):
	img = processImage(imagePath)
	output = model(img)
	#output = postFunc(output)
	print(output)

	prob,ind = torch.max(output, 1)
	pred = int(ind.cpu().numpy())
	return output,pred


def loadModel(modelPath):
	model = Xception()

	device = 'cpu'
	if torch.cuda.is_available():
		model.cuda()
		device = 'cuda:0'

	model.load_state_dict(torch.load(modelPath, torch.device(device)))
	return model


def getDataset(datasetPath, data):
	filePaths = []
	tags = []
	if datasetPath[-1] != '/':
		datasetPath += '/'
	for sequences in os.listdir(datasetPath):
		if data == 'faceForensics' and 'sequences' in sequences:
			for faceTools in os.listdir(datasetPath+sequences):
				for video in os.listdir(datasetPath+sequences+'/'+faceTools+'/c23/videos/'):
					videoPath = datasetPath+sequences+'/'+faceTools+'/c23/videos/'+video
					filePaths.append(videoPath)
					if 'original' in videoPath:
						tags.append(0)
					elif 'manipulated' in videoPath:
						tags.append(1)

		elif data == 'GAN' and 'GAN' in sequences:
			for tag in os.listdir(datasetPath+sequences):
				for imagePath in os.listdir(datasetPath+sequences+'/'+tag):
					imagePath = datasetPath+sequences+'/'+tag+'/'+imagePath
					filePaths.append(imagePath)
					if 'real' in tag:
						tags.append(0)
					elif 'fake' in tag:
						tags.append(1)
	
	# shuffle data
	temp = list(zip(filePaths, tags))
	shuffle(temp)
	filePaths, tags = zip(*temp)

	return filePaths, tags

def main():
	dataset = 'faceForensics'
	bSize = 10
	model = Xception()
	if torch.cuda.is_available():
		model.cuda()
	filePaths, tags = getDataset('dataset/', data=dataset)
	trainFullNetwork(model, filePaths, tags, batchSize=bSize, lr=0.05, epochs=3, data=dataset)
	'''

	convertVideoToImages('fake.mp4', 'fake/')
	convertVideoToImages('real.mp4', 'real/')
	model = loadModel('faceForensics_model.pth')

	correct = 0
	total = len(os.listdir('fake/'))
	for img in os.listdir('fake/'):
		img = 'fake/'+img
		prob,pred = predict(model, img)
		os.remove(img)

		label = 'fake' if pred == 1 else 'real'

		#print('Image is '+label)
		#print('Probability: '+str(prob))
		if label == 'fake':
			correct+=1

	print('Fake')
	print(str(correct)+'/'+str(total))
	print(correct/total)
	print('\n\n')

	correct = 0
	total = len(os.listdir('real/'))
	for img in os.listdir('real/'):
		img = 'real/'+img
		prob,pred = predict(model, img)
		os.remove(img)

		label = 'fake' if pred == 1 else 'real'

		#print('Image is '+label)
		#print('Probability: '+str(prob))
		if label == 'real':
			correct+=1

	print('Real')
	print(str(correct)+'/'+str(total))
	print(correct/total)
	'''

if __name__ == "__main__":
	main()
