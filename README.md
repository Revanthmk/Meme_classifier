# Noyoto_meme_classifier
A model for classifying whether an image is meme or not

# Problem Statement
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

# Files
noyoto.ipynb 
> Contains the whole process of data selection, data exploration, image augmentation, benchmark model training, final model training, model evaluation, model inference, exporting the model.
>
final101_2classes.h5
> Final tensorflow model file, used for inference.
>
inference.py
> Inference script
>
meme.jpg
> Sample inference image of class 0
>
non_meme.jpg
> Sample inference image of class 1
>

# Dataset Selection
The problem statement requires a dataset with 2 classes, memes and images which are not memes.
## Discovered datasets
### sayangoswami/reddit-memes-dataset
```
This dataset contains 3326 memes.
Initial inspection of the dataset doesn't look good,
the dataset contains memes but most of the memes are just funny images like
> image of a fish in sand
> A kid jumping out of a bed
> Google screenshot of a potato.
We can't use this dataset because it is too similar to non-meme images.
```

### parthplc/facebook-hateful-meme-dataset
```
This dataset contains 10,000 images of memes.
Initial inspection of the dataset seems promising, the dataset contains memes in the traditional sense conatining image over text in most of the data points.
We are going on with this dataset.
```

# Downloading the dataset
We can download and unzip the dataset from Kaggle using kaggle-api
```
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
pip install -q kaggle
kaggle datasets download -d parthplc/facebook-hateful-meme-dataset
unzip 'facebook-hateful-meme-dataset.zip' -d 'data/'
```










