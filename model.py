#!/usr/bin/python3
import sys
import os
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch
from PIL import Image
import cv2
import dlib
from videoToImages import convertVideoToImages


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


def processImage(imagePath):
	print("...processing image")
	print("Image: "+imagePath)
	
	img = cv2.imread(imagePath)
	img = getFaceCrop(img)
	process = transforms.Compose([
			transforms.Resize((299, 299)),
			transforms.ToTensor(),
			transforms.Normalize([0.5]*3, [0.5]*3)
		])
	processedImg = process(Image.fromarray(img))
	processedImg = processedImg.unsqueeze(0)
	if torch.cuda.is_available():
		processedImg = processedImg.cuda()
	return processedImg


def trainModel(model, datasetPath, faceForensics=False, personDoesNotExist=False, catDoesNotExist=False):
	# make all layers trainable
	for i, param in model.named_parameters():
		param.requires_grad = True

	if datasetPath[-1] != '/':
		datasetPath += '/'
	for sequences in os.listdir(datasetPath):
		if faceForensics and 'sequences' in sequences:
			for faceTools in os.listdir(datasetPath+sequences):
				for video in os.listdir(datasetPath+sequences+'/'+faceTools+'/c23/videos/'):
					videoPath = datasetPath+sequences+'/'+faceTools+'/c23/videos/'+video
					
					imagesDir = videoPath.replace('videos', 'images')
					imagesDir = imagesDir.replace('.mp4', '/')

					if not os.path.exists(imagesDir):
						os.makedirs(imagesDir)
						convertVideoToImages(videoPath, imagesDir)

					for imagePath in os.listdir(imagesDir):
						imagePath = imagesDir + imagePath
						img = processImage(imagePath)
						outputs = model(img)
						print(outputs)

						# delete image
						os.remove(imagePath)

					# delete empty directory
					os.rmdir(imagesDir)
		elif personDoesNotExist and 'thispersondoesnotexist' in sequences:
			for image in os.listdir(datasetPath+sequences):
				imagePath = datasetPath+sequences+'/'+image
				img = processImage(imagePath)
				outputs = model(img)
				print(outputs)
		elif catDoesNotExist and 'thiscatdoesnotexist' in sequences:
			for image in os.listdir(datasetPath+sequences):
				imagePath = datasetPath+sequences+'/'+image
				img = processImage(imagePath)
				outputs = model(img)
				print(outputs)

		# save model
		modelPath = ''
		if faceForensics:
			modelPath += 'faceForensics'
		if personDoesNotExist:
			if modelPath == '':
				modelPath += '_'
			modelPath += 'personDoesNotExist'
		if catDoesNotExist:
			if modelPath == '':
				modelPath += '_'
			modelPath += 'catDoesNotExist'
		modelPath += '_model.pth'

		torch.save(model.state_dict(), modelPath)


def predict(model, imagePath):
	img = processImage(imagePath)
	output = model(img)

	prob,ind = torch.max(output, 1)
	pred = int(ind.cpu().numpy())
	return prob,pred


def loadModel(modelPath):
	model = Xception()

	device = 'cpu'
	if torch.cuda.is_available():
		model.cuda()
		device = 'cuda:0'

	model.load_state_dict(torch.load(modelPath, torch.device(device)))
	
	return model

def main():
	
	model = Xception()
	if torch.cuda.is_available():
		model.cuda()
	trainModel(model, 'dataset/', faceForensics=True)
	'''

	model = loadModel('faceForensics_model.pth')
	prob,pred = predict(model, '11.jpeg')

	label = 'fake' if pred == 1 else 'real'

	print('Image is '+label)
	print('Probability: '+str(prob))
	'''

if __name__ == "__main__":
	main()
