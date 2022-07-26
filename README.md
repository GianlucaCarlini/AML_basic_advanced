# Automatic segmentation of face masks with Convolutional Neural Networks

Project for the AML PhD Course (both basic and advanced).

The main code of the project can be found in model.ipynb. The explanation of the other python scripts used in the project can also be found in the model notebook. Briefly:
* **train_test_split.py** is used to split the data in mutually exclusive subsets.
* **patch_generator.py** is used to generate patches of arbitrary dimension from the original images.
* **Utils** contains utility codes to generate the segmentation model and to calculate the metrics and loss function used in the work.

The original data are not included in this repository since they can be found [here](https://www.kaggle.com/datasets/perke986/face-mask-segmentation-dataset), and the provided scripts (**train_test_split.py** and **patch_generator.py**) can be used to obtain the same exact splitting of the data and image patches used for this work.
