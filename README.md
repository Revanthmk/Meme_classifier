# 1) Noyoto_meme_classifier
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
> Requirements file
>
# 4) Dependencies
```
- Python 
- opencv-python
- numpy
- tensorflow
```
Note that the dependencies mentioned here and in the requirements.txt file are only for running the inference and NOT for training the model.
# 5) Run Inference
```
- pip install -r requirements.txt
- Upload the inference images in 'inf_input' directory.
- Install the python dependencies from requirements.txt
- Run the inference.py file.
- Results will be created in the 'inf_output' directory.
- Wait till the inference is completed.
- You can find the output in 'inf_output' directory containing the same images with additional subtext displayed on the top-left indication whether the image is classified as 'meme' or 'non=meme'.
```
# 6) Model Performance
![image](https://github.com/Revanthmk/Noyoto_meme_classifier/assets/38763740/0ad9ff86-6592-44f4-acb2-df724508cfbb)
![image](https://github.com/Revanthmk/Noyoto_meme_classifier/assets/38763740/ca7a95da-1248-493e-b196-bd024806834c)



