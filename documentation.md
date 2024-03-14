# 1) Meme_classifier
A model for classifying whether an image is meme or not

# 2) Problem Statement
```
Assignment

Please find the assignment below.
Build a model to classify whether an image is meme or not
You are free to
● Use any open source libraries, if required
● Use publicly available datasets to train the model
● Use any programming language, but python is preferred

You are expected to
● Provide evaluation and performance metrics for your model
● Create a readme file with instructions on how to run the test cases
● Upload your code on GitHub and email repo link to #########
```
# 3) Files
documentation.md
> MD file explaining the whole process of the project
> 
noyoto.ipynb 
> Contains the whole process of data selection, data exploration, image augmentation, benchmark model training, final model training, model evaluation, model inference, exporting the model.
>
final101_2classes.h5
> Final tensorflow model file, used for inference.
>
inference.py
> Inference script
>
inf_input/meme.jpg
> Sample inference image of class 0
>
inf_input/non_meme.jpg
> Sample inference image of class 1
>
requirements.txt
> Requirements file to run to get all the python dependencies
>
# 4) Dataset Selection
The problem statement requires a dataset of images containing memes.
## a) Discovered datasets
### i) sayangoswami/reddit-memes-dataset
```
This dataset contains 3326 memes.
Initial inspection of the dataset doesn't look good,
the dataset contains memes but most of the memes are just funny images like
> image of a fish in sand
> A kid jumping out of a bed
> Google screenshot of a potato.
We can't use this dataset because it is too similar to non-meme images.
```
### ii) parthplc/facebook-hateful-meme-dataset
```
This dataset contains 10,000 images of memes.
Initial inspection of the dataset seems promising, the dataset contains memes in the traditional sense conatining image over text in most of the data points.
We are going on with this dataset.
```
# 5) Downloading the dataset
We can download and unzip the dataset from Kaggle using kaggle-api
```
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
pip install -q kaggle
kaggle datasets download -d parthplc/facebook-hateful-meme-dataset
unzip 'facebook-hateful-meme-dataset.zip' -d 'data/'
```
# 6) Data Analysis
```
The dataset has 10,000 data points.
Each data point is an image and a corresponding json file containing the text in the image.
We are going to ignore the text in the json file and only going to work with the image hence to generalize the model to work on any meme, and not overfit(in this specific usecase) to hateful memes.
```
# 7) Visualizing the Dataset
```
Example data point : ![01264](https://github.com/Revanthmk/Noyoto_meme_classifier/assets/38763740/a26c44d2-2f3d-47db-9771-c62a3c4a1bbb)
The data contains images mostly of this format, image with embedded text.
Which corresponds to a meme in general.
```
# 8) Data Preprocessing
```
We are using the following preprocessing steps.
Hyperparameter selection for each of these preprocessing methods are explained in the noyoto.ipynb notebook with examples
1) ZCA Whitening : True
   -

2) ZCA Epsilon : 1e-06
   -

3) Rotation Range : 10
   - +- 10 degree rotation

4) Width Shift Range : 0.3
   - 30% width shift

5) Height Shift Range : 0.3
   - 30% height shift

6) Brightness Range : [0,2]
   - If the meme is a screenshot from dark mode of an app

7) Shear Range : 30
   - More than 30% shearing can illegitimate the text

8) Zoom Range : [0.2, 2]
   - 0.2 to 2 zoom out and zoom in range should be enough for memes

9) Channel Shift Range : 80
   - Same meme can be made in different colors

10) Fill Mode : Nearest
    - Nearest mode will just expand the image instead of adding a constant color at the ends which can overfit the model

11) Horizontal Flip : False
    - No need of horizontal flip in memes because the text will be reversed

12) Vertical Flip : False
    - No need of horizontal flip in memes

13) Rescale : 1./255
    - rescaling for avoiding vanishing gradients or gradient explosions and reduced memory requirements.
```
# 9) Benchmark CNN
```
Building a benchmark CNN model to
- Checking if everything else except the model architecture works as expected syntactically
- Establishing a baseline performance
```
### a) Architecture
![image](https://github.com/Revanthmk/Noyoto_meme_classifier/assets/38763740/15cf4708-a27a-4666-ab25-188e1acaab91)
### b) Evaluation  
```
- Evaluation of this model doesn't make any sense because we are training a classification architecture model with just one class
```
### c) Observation
```
- The dataset has only 1 class and is more suitable for anomaly detection models like Autoencoders, but the problem statement mentioned to build a classification model
- So the next step is to find a dataset with 2 classes, one with meme and one without memes.
- Going back to dataset selection process again
```
# 10) Dataset Selection
```
Couldn't find any dataset with this specific requirements with 2 classes.
Next step will be to build a dataset.
```
## a) Building the Dataset
```
- We need 2 classes memes and non-memes.
- We already have 10,000 images of memes.
- Should look for non-meme images.
- Best source of non-meme images will be the ImageNet dataset.
```
#### i) Building non-meme class
```
- We need a second non-meme class to build a classification model,
Decided to choose famous 10 classes from the ImageNet dataset and choose 30 images from each class and tag it as non-meme class
There are few different methods we can create the dataset.
1) Get 200 images from the each of the top 10 classes of ImageNet dataset, but the classes in ImageNet are too specific.
2) Get the testing or validation images from ImageNet dataset, this should have enough variation in the images to serve as our non-meme class.
3) The downloading process of ImageNet dataset can be found in the noyoto.ipynb file.
4) Now we have 2 classes one with 10,000 memes and another with 3326 non-meme images.
5) Same preprocessing steps are followed for this dataset.
6) Now again training to the benchmark model with this new dataset.
```
# 11) Benchmark CNN
```
Building a benchmark CNN model to
- Checking if everything else except the model architecture works as expected syntactically
- Establishing a baseline performance
```
### a) Architecture
![image](https://github.com/Revanthmk/Noyoto_meme_classifier/assets/38763740/476bd836-7482-49b7-8f3d-7fb22f504ec4)
### b) Evaluation
![image](https://github.com/Revanthmk/Noyoto_meme_classifier/assets/38763740/385ee5f1-23aa-4bfa-8673-ba8758701a82)
![image](https://github.com/Revanthmk/Noyoto_meme_classifier/assets/38763740/8b2d4c25-6bf2-466f-a1c2-38d42f19fa90)
### c) Observation
```
- We have trained the model only for 5 epochs because of restrictions in GPU resources.
- The Validation accuracy of the model is 75%.
- Recall for class 0 is 1 and for that of class 1 is 0.
- Precision for class 0 is 0.75 and for that of class 1 is 0.
- With the 2 above statements and given that we have 75% of our total data as training data, we can conclude that the model is predicting every image as class 0.
```
# 12) ResNet 50 (transfer learning)
```
- Looking at the amount of data we have, transfer learning will be the best approach instead of training the CNN from scratch.
- Going with ResNet pretrained with ImageNet dataset and training the model with transfer learning.
- Reasons for choosing ResNet pretrained with ImageNet include,
  1) The memes and non-memes classes look similar to the ImageNet dataset, so we can use the pre-trained feature extraction layers from ResNet
  2) Skip connections in ResNet will accelerate the training process.
  3) Deeper Architecture with 50 and 101 layers lets the model learn more abstract features.
  4) Freezing everything except the last 2 layers should give enough features for our training process to work with
  5) Adding a GlobalAveragePooling layer, Dense layer with 128 neurons, BatchNormalization layer and finally the output layer with 2 neurons for our 2 classes.
  6) Excluded the Dropout layer because BatchNormalization provides the same regularization as Dropout.
```
### a) Architecture
![image](https://github.com/Revanthmk/Noyoto_meme_classifier/assets/38763740/17f80cf4-0ede-45bd-9679-6ecac238c67b)
### b) Evaluation
![image](https://github.com/Revanthmk/Noyoto_meme_classifier/assets/38763740/85d37596-7451-4c7c-ac30-0c4b5885b00b)
![image](https://github.com/Revanthmk/Noyoto_meme_classifier/assets/38763740/b03a4bed-1178-4acd-a60d-c0bec6bcaa0d)
### c) Observation
```
- We have trained the model only for 5 epochs because of restrictions in GPU resources.
- The validation accuracy of the model is 61%.
- Recall for class 0 is 0.73 and for that of class 1 is 0.27.
- Precision for class 0 is 0.75 and for that of class 1 is 0.25.
- F1 score for class 0 is 0.74 and for that of class 1 is 0.26.
- The model performs reasonably well for the class 0 and poor for class 1.
- The above result is expected because we trained our model only for 5 epochs which is too low for any problem statement.
```
# 13) ResNet 101 (transfer learning)
```
- All the above points for ResNet 50 would also go for ResNet 101.
- Trying out 101 layer model will lead to better accuracy at the expence of inference speed and training speed.
- The problem statement didn't mention any restrictions on the inference speed or any cap in the training resource, so there is not many reasons against using ResNet 101 when we get better accuracies out of the model.
- The only problem we might face using a deeper model is overfitting, but that is resolved by our extensive image augmentation methods, regularization techniques like Batch Normalization layers.
```
### a) Architecture
![image](https://github.com/Revanthmk/Noyoto_meme_classifier/assets/38763740/5760a66f-a7e7-4257-8670-72e2ca469a17)
### b) Evaluation
![image](https://github.com/Revanthmk/Noyoto_meme_classifier/assets/38763740/0ad9ff86-6592-44f4-acb2-df724508cfbb)
![image](https://github.com/Revanthmk/Noyoto_meme_classifier/assets/38763740/ca7a95da-1248-493e-b196-bd024806834c)
### c) Observation
```
- The validation accuracy of the model is 66%
- Recall for class 0 is 0.81 and for that of class 1 is 0.19.
- Precision for class 0 is 0.75 and for that of class 1 is 0.25.
- F1 score for class 0 is 0.78 and for that of class 1 is 0.21.
- The model seems to do better than our previous models in the class 0 but fails in class 1, which can again be because of the number of epochs the model is trained on.
- The reason for the model performing significantly better in one class might be because of the class imbalance, the next step would be the add more data points to class 1
```
# 14) Improvements
```
- The main and most important improvement which can be done to the model is training the model for multiple more number of epochs.
- Adding more data to the non-meme class to match the meme class should balance the dataset.
```



