# Generated Image Detection

Implementation of the algorithm described in [FaceForensics++: Learning to Detect Manipulated Facial Images](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf) with a focus on the XceptionNet classifier. 

## Description

Synthetic image generation is becoming more advanced to the point where it is becoming increasingly difficult for people to recognize fake, generated images. The more common these high quality but fake generated images become, the more harm they can cause by spreading false information. And this has also caused a loss of trust in digital content. There are also many different methods of generating synthetic images, especially when it comes to facial manipulation methods. The [FaceForensics++](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf) dataset includes original videos as well as the altered versions of those videos using many different manipulation methods. We implemented our own version of the algorithm described in [FaceForensics++: Learning to Detect Manipulated Facial Images](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf) with a focus on the XceptionNet classifier. We used a subset of the [FaceForensics++](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf) dataset in order to initially train and test the model. After using the FaceForensics++ dataset, we decided to additionally test the classifier using our own dataset of GAN images obtained from [thispersondoesnotexist.com](https://thispersondoesnotexist.com/).

## Dataset

### FaceForensics Dataset

The FaceForensics++ dataset is an extensive dataset of original YouTube videos, original actor videos, and altered versions of those videos. The entire dataset requires 1.6TB of storage space. Due to the large size of the dataset, we have opted to use a subset. For training, we used a total of 1000 videos. 200 original Youtube videos, 100 of the YouTube videos altered with the different methods for a total of 500 altered videos, 200 original actor videos, and 100 of those actor videos altered with DeepFakes. Altogether, the training dataset had 700 real and fake YouTube videos and 300 real and fake actor videos. For the testing dataset, we used the same video distribution as for the training dataset but only one tenth the amount of videos. Meaning that we used 20 original Youtube videos, 10 of the YouTube videos altered with the different methods for a total of 50 altered videos, 20 original actor videos, and 10 of those actor videos altered with DeepFakes. Altogether, the training dataset had 70 real and fake YouTube videos and 30 real and fake actor videos for a total of 100 videos.

#### Generating Frames from Videos

Video classification involves looking at each frame within the video individually, meaning that what is classified is a series of images. We implemented a Python script to take the dataset videos one at a time and convert them into frames using ffmpeg. The video frames are created with the naming convention \<original video name\>-\<frame number\>.jpg. Immediately after training the model with all the frames from a video, the frames are deleted to save space.

### GAN Image Dataset - [thispersondoesnotexist.com](https://thispersondoesnotexist.com/)

We collected our own dataset of GAN generated images by downloading 1500 images from thispersondoesnotexist.com. The website thispersondoesnotexist.com generates a completely new image of a person every time the web page is loaded. The images are created using the style-based GAN architecture StyleGAN.

To accurately test these GAN images against real images, we used additional datasets of real images that were cropped in a similar fashion to the GAN images. The datasets are from [kaggle.com](https://www.kaggle.com/ciplab/real-and-fake-face-detection) and [NVlabs](https://github.com/NVlabs/ffhq-dataset).

Our training dataset included 1000 GAN images from thispersondoesnotexist.com, 800 real images from the Real and Fake Face Detection dataset, and 200 real images from NVlabs for a total of 2000 training images. For testing, we used 500 GAN Images, 250 from the Real and Fake Face Detection dataset, and 250 from NVlabs for a total of 1000 testing images.

## Usage

### Included Files

* `faceforensics-download.py`
  * Script provided to download videos from the FaceForensics++ dataset. Documentation can be found on the [FaceForensics Dataset GitHub repo](https://github.com/ondyari/FaceForensics/tree/master/dataset#1-download-script).
* `fetchGeneratedImages.py`
  * Script to download images from [thispersondoesnotexist.com](https://thispersondoesnotexist.com/). To run this script, run `python3 fetchGeneratedImages.py`. The script is setup to download 200 images and store them in a folder named `dataset`.
* `videoToImages.py`
  * The code in this file is intended to be called by `model.py` so that the video that is currently being evaluated can be converted into images.
* `model.py`
  * This file contains the XceptionNet model implementation, training, and testing modules. Instructions can be found under [Using the Model](#using-the-model)
* `trainConfig.json`
  * This is a configuration file for specifying the training hyperparameters. Adjust values as needed.
* `requirements.txt`
  * This file contains packages needed to train and/or test the model. Run `pip install -r requirements.txt` to install these dependencies.

### Using the Model

#### Training
For training the model, run `model.py` as shown below:

```bash
$ python3 model.py train <modelType> <datasetType> <datasetPath>
```

`modelType` options include `XceptionNet` and `ResNet`
`datasetType` options include `faceForensics` and `GAN`
`datasetPath` is used to specify the path to the training dataset

Example:

```bash
$ python3 model.py train XceptionNet faceForensics dataset/
```

Training hyperparameters such as batch size, learning rate, weight decay, and epochs can be adjusted in `trainConfig.json`.

The model will automatically be saved on local memory as a `.pth` file once training is completed


#### Testing

For testing the model, run `model.py` as shown below:

```bash
$ python3 model.py test <modelType> <datasetType> <datasetPath> <modelPath>
```

`modelType` options include `XceptionNet` and `ResNet`
`datasetType` options include `faceForensics` and `GAN`
`datasetPath` is used to specify the path to the testing dataset
`modelPath` is used to specify the path of the trained model

Example:

```bash
$ python3 model.py test XceptionNet faceForensics dataset/ faceForensics_XceptionNet.pth
```

Evaluations such as accuracy, precision, recall, and F1 score will be computed and printed once testing is completed


#### Dataset Structure

`sampleDatasetStructure` provided to show how the dataset is structured. The tree below shows the structure of this directory. The FaceForensics++ dataset is made up of `manipulated_sequences` and `original_sequences`.

```
sampleDatasetStructure
├── GAN
│   ├── fake_thispersondoesnotexist
│   └── real
├── manipulated_sequences
│   ├── DeepFakeDetection
│   │   └── c23
│   │       └── videos
│   ├── Deepfakes
│   │   └── c23
│   │       └── videos
│   ├── Face2Face
│   │   └── c23
│   │       └── videos
│   ├── FaceShifter
│   │   └── c23
│   │       └── videos
│   ├── FaceSwap
│   │   └── c23
│   │       └── videos
│   └── NeuralTextures
│       └── c23
│           └── videos
└── original_sequences
    ├── actors
    │   └── c23
    │       └── videos
    └── youtube
        └── c23
            └── videos
```