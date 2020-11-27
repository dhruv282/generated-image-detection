# Generated Image Detection

Implementation of the algorithm described in [FaceForensics++: Learning to Detect Manipulated Facial Images](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf) with a focus on the XceptionNet classifier. 

## Description

Synthetic image generation is becoming more advanced to the point where it is becoming increasingly difficult for people to recognize fake, generated images. The more common these high quality but fake generated images become, the more harm they can cause by spreading false information. And this has also caused a loss of trust in digital content. There are also many different methods of generating synthetic images, especially when it comes to facial manipulation methods. The [FaceForensics++](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf) dataset includes original videos as well as the altered versions of those videos using many different manipulation methods. We implemented our own version of the algorithm described in [FaceForensics++: Learning to Detect Manipulated Facial Images](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf) with a focus on the XceptionNet classifier. We used a subset of the [FaceForensics++](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf) dataset in order to initially train and test the model. After using the FaceForensics++ dataset, we decided to additionally test the classifier using our own dataset of GAN images obtained from [thispersondoesnotexist.com](https://thispersondoesnotexist.com/).

## Dataset

### FaceForensics Dataset

<Paraphrase some of their description, describe our subset>

#### Generating Frames from Videos

### GAN Image Dataset

[This Person Does Not Exist](https://thispersondoesnotexist.com/)

## Using the Pre-trained Model
