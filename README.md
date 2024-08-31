# Binary Classification Model Using Support Vector Machine (SVM) and Neural Network
=====================================================================

Table of Contents
-----------------

1. [Project Title and Description](#project-title-and-description)
2. [Installation Instructions](#installation-instructions)
3. [Usage Examples](#usage-examples)
4. [Detailed Model Architecture and Hyperparameter Tuning Explanation for SVM](#detailed-model-architecture-and-hyperparameter-tuning-explanation-for-svm)
5. [Detailed Model Architecture and Hyperparameter Tuning Explanation for Neural Network](#detailed-model-architecture-and-hyperparameter-tuning-explanation-for-neural-network)
6. [Model Evaluation Metrics](#model-evaluation-metrics)

## Project Title and Description
-------------------------------

This project involves developing two binary classification models using a Support Vector Machine (SVM) and a neural network architecture with hyperparameter tuning capabilities.

Binary classification is a fundamental problem in machine learning where the goal is to predict one of two classes or categories. This project aims to create efficient and accurate binary 
classification models that can be used for various real-world applications.

## Installation Instructions
---------------------------

To use this code, you will need to have the following libraries installed:

* `numpy`
* `pandas`
* `scikit-learn`
* `keras` (with a compatible backend like TensorFlow or Theano)
* `gridsearchcv`

You can install these libraries using pip, the Python package manager. Here's how you can do it:

```bash
pip install numpy pandas scikit-learn keras gridsearchcv
```

## Usage Examples
-----------------

To use this code, follow these steps:

1. Clone the repository to your local machine.
2. Open a terminal and navigate to the project directory.
3. Run the script `svm.py` or `nn.py` depending on which model you want to train.
4. The script will perform hyperparameter tuning using GridSearchCV and print out the best parameters found.

## Detailed Model Architecture and Hyperparameter Tuning Explanation for SVM
--------------------------------------------------------------------------------

The SVM model uses a linear kernel and performs binary classification on the given data. The hyperparameters tuned are:

* `kernel`: 'linear', 'poly', or 'rbf'
* `degree`: 1, 2, or 3
* `gamma`: 'scale' or 'auto'
* `C`: 0.1, 1, or 10

The GridSearchCV algorithm is used to perform hyperparameter tuning over a grid of values specified above.

## Detailed Model Architecture and Hyperparameter Tuning Explanation for Neural Network
------------------------------------------------------------------------------------

The neural network model uses three dense layers with a total of 16 neurons in each layer (except for the output layer, which has one neuron). The activation function used is 'sigmoid' for all 
layers except for the output layer. The hyperparameters tuned are:

* `activation`: 'sigmoid' or 'tanh'
* `batch_size`: 10, 50, or 100
* `epochs`: 5, 10, or 20
* `loss`: 'binary_crossentropy', 'squared_hinge', or 'kullback_leibler_divergence'
* `optimizer`: 'adam' or 'rmsprop'

The GridSearchCV algorithm is used to perform hyperparameter tuning over a grid of values specified above.

## Model Evaluation Metrics
----------------------------------------------------------------

Both models are evaluated using several metrics:

* Brier score (probability calibration)
* Precision-recall curve and PR AUC
* F2 score
