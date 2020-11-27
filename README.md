# Generated Image Detection

Implementation of the algorithm described in [FaceForensics++: Learning to Detect Manipulated Facial Images](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf) with a focus on the XceptionNet classifier. 

## Description

Synthetic image generation is becoming more advanced to the point where it is becoming increasingly difficult for people to recognize fake, generated images. The more common these high quality but fake generated images become, the more harm they can cause by spreading false information. And this has also caused a loss of trust in digital content. There are also many different methods of generating synthetic images, especially when it comes to facial manipulation methods. The [FaceForensics++](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf) dataset includes original videos as well as the altered versions of those videos using many different manipulation methods. We implemented our own version of the algorithm described in [FaceForensics++: Learning to Detect Manipulated Facial Images](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf) with a focus on the XceptionNet classifier. We used a subset of the [FaceForensics++](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf) dataset in order to initially train and test the model. After using the FaceForensics++ dataset, we decided to additionally test the classifier using our own dataset of GAN images obtained from [thispersondoesnotexist.com](https://thispersondoesnotexist.com/).

## Dataset

### FaceForensics Dataset

The FaceForensics++ dataset is an extensive dataset of original YouTube videos, original actor videos, and altered versions of those videos. The entire dataset requires 1.6TB of storage space. Due to the large size of the dataset, we have opted to use a subset. For our training dataset... <to be filled in>. For the testing dataset, we used a very small subset. We used 10 original YouTube videos as well as those same videos altered with each of the 5 manipulation methods, totaling 60 videos. Additionally, we used 10 original actor videos and those same 10 videos altered with DeepFakes for another 20 videos. In total, our testing dataset consisted of 80 videos.

#### Generating Frames from Videos

Video classification involves looking at each frame within the video individually, meaning that what is classified is a series of images. We implemented a Python script to take the dataset videos one at a time and convert them into frames using ffmpeg. The video frames are created with the naming convention <original video name>-<frame number>.jpg. Immediately after training the model with all the frames from a video, the frames are deleted to save space.

### GAN Image Dataset - [thispersondoesnotexist.com](https://thispersondoesnotexist.com/)

We collected our own dataset of GAN generated images by downloading 200 images from thispersondoesnotexist.com. The website thispersondoesnotexist.com generates a completely new image of a person every time the web page is loaded. The images are created using the style-based GAN architecture StyleGAN.

To accurately test these GAN images against real images, we used an additional dataset of real images that were cropped in a similar fashion to the GAN images. The dataset is from [kaggle.com](https://www.kaggle.com/ciplab/real-and-fake-face-detection).

## Using the Pre-trained Model
