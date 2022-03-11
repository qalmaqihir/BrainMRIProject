# Brain MRI Classification

> This repo contains all the code base for my final year project. The Brain MRI Classification. This project aims to build an accurate and robust ML model for classification of normal and 
abnormal brain MRI images. The basic idea is to use Image Processing as well as Machine Learning Algorithms to implment a novel and improved model; which surpasses other state of the art
models in literature.
> This repository has jupyter notebooks - providing detail notes on the topics - as well as MATLAB codes for implementing Image Processing algorithms to evaluate the best ones for the actual model.


## Contents 
### 1. Image Processing (Jupyter notebooks)
    - Basic Conversions, Reading and writing to images [chpt1-3]
    - Image Filters and Image Enchancement (transformations) [chpt4-5] 
    - Filters preprocessing
    - Transforms preprocessing
    - Feature Extractions
### 2. MATLAB Code
    - Basic Image handling, Filters and Feature Extraction
    - Medain, Skewness, Standard Deviation, Kurtosis, Entropy
### 3. Models
    - Artificial Neurla Network
            - Red, Green, Blue channel saved model
            - Confusion matrix for testing and Training
            - model 
    - Decision Tree 
            - Red, Green, Blue channel saved model
            - Confusion matrix for testing and Training
            - model 
    
    - Support Vector Machine
    - Random Forest
            - Red, Green, Blue channel saved model
            - Confusion matrix for testing and Training
            - model 
    -
    
### 4. Performance Evaluation
    - Confusion matrix
    - F1-score, Accuracy, Precision and Recall were calculated.
    - Performance is evaluated on output from majority vote and the y_test
    -
    
### 5. Dataset
    - The Dataset is three files (red, green and blue) in excel as well as csv formats. And a Labels file with the classes.
    - For the model, we gave all three files to three classifiers alone with the labels file
    - All the algorithms were give the same data files
    -


## Installation

OS X & Linux:

```sh
git clone https://github.com/qalmaqihir/BrainMRIProject.git
```

## Usage example
Each Jupyter notebook contains sufficeint information about the topics covered. While the MATLAB code has comments to guide the reader about the processes.

_For more about each topic and the model, please refer to the [Wiki][wiki]._

## Development setup
-
-
-

## Release History
Each commit has its own history...

## Meta
Jawad Haider – [@JawadHa49605912](https://twitter.com/JawadHa49605912?t=LImgqrvKUy48gqaaeKooBA&s=09) – jawad.haider_2022@ucentralasia.org

_Distributed under the MIT license. See ``LICENSE`` for more information or [check here][LICENSE]._

## Contributing

1. Fork it (<https://github.com/qalmaqihir/BrainMRIProject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[wiki]: https://github.com/qalmaqihir/BrainMRIProject/wiki
[LICENSE]: https://github.com/qalmaqihir/BrainMRIProject/blob/main/LICENSE
