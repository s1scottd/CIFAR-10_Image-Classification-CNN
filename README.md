# CIFAR-10 Image Classification with CNN

Convolutional Neural Network (CNN) model for image classification on the CIFAR-10 dataset.

## Table of Contents

1. [Introduction](#introduction)
2. [Technologies](#technologies)
3. [Installation](#installation)
4. [Data](#data)
5. [Methodology](#methodology)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [References](#references)

## Introduction <a name="introduction"></a>

This project involves building and training a Convolutional Neural Network, both without autotuning and with autotuning, to classify images from the CIFAR-10 dataset.

## Technologies <a name="technologies"></a>

The project uses the following technologies and libraries:

- Python:
- Jupyter Notebook
- TensorFlow:
- Keras:
- NumPy:
- KerasTuner:

## Installation <a name="installation"></a>

Follow these steps to set up the project:

1. Clone the repository: `git clone https://github.com/s1scottd/CIFAR-10_classification.git`
2. Open Google Colab and upload the notebook file `CIFAR-10_classification.ipynb` or select the colab button in the notebook.

## Data <a name="data"></a>

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 40000 training images, 10000 validation images and 10000 test images.

## Methodology <a name="methodology"></a>

The project involves the following steps:

1. Load and preprocess the data: The CIFAR-10 data is loaded using Keras and normalized to have pixel values between 0 and 1.
2. Define the CNN model: A CNN model is defined using Keras with two convolutional layers, a max pooling layer, and two dense layers.
3. Compile and train the model: The model is compiled and trained using the Adam optimizer and categorical cross-entropy loss function for 20 epochs.
4. Evaluate the model: The model's performance is evaluated on the test data using accuracy as the metric.
5. Define the CNN model: A CNN model is defined using hyperparameter.
6. Compile and train the model: The model is trained repeatedly using an autotuner.
7. Evaluate the best model: The model's performance is evaluated on the test data using accuracy as the metric.

## Results <a name="results"></a>

The trained model achieves an accuracy of xx% on the test data. For a more detailed view of the model's performance and visualizations, check the Jupyter notebook `CIFAR-10_Classification.ipynb`.

## Conclusion <a name="conclusion"></a>

This project demonstrates the effectiveness of Convolutional Neural Networks in image classification tasks. Future work might involve exploring different model architectures, or using data augmentation techniques.

## References <a name="references"></a>

- O'Malley, Tom and Bursztein, Elie and Long, James and Chollet, Fran\c{c}ois and Jin, Haifeng and Invernizzi, Luca and others. KerasTuner, github.com/keras-team/keras-tuner. 
- Chollet, F. et al., 2015. Keras.
