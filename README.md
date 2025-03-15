Lung-Colon Cancer Classification based on Custom CNN Model.

A Deep Learning model designed to distinguish between different types of lung and colon cancer histopathological images  ([Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)) 

The dataset consists of 5 Classes:
1. Colon Adenocarcinoma (colon_aca), 
2. Normal Colon Tissue (colon_n),
3. Lung Adenocarcinoma (lung_aca),
4. Normal Lung Tissue (lung_n)
5. Lung Squamous Cell_Carcinoma (lung_scc).

This repository contains a pre-trained model, a Jupyter notebook demonstrating the model preparation process, and a Gradio-powered web interface for testing the model.

## Table of Contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Features](#features)
- [Installation](#installation)
- [Model Info](#model-training)
- [Web Interface](#web-interface)
- [Conclusion](#conclusion)

## Introduction
This project utilizes the PyTorch framework and a Convolutional Neural Network (CNN) model to classify histopathological images of lung and colon cancer across five different classes. The model is trained to distinguish between different types of lung and colon tissue samples, leveraging deep learning for accurate classification.

## Usage
- Download the folder and make sure to install all libraries from the requirements file.
- Ensure that the pre-trained [`LungDisease_Classfier`](LungDisease_Classifier.pth) file is in the project directory
- Run the [`classify.py`](classify.py) file to launch the web-interface and upload image to classify it's class.
   
## Features
- Custom pre-trained model for quick use on  lung and colon cancer classification task.
- Jupyter notebook to demonstrate the whole process of model training and testing using pytorch.
- Gradio web interface for easy testing and interaction.
- Lightweight model without any quantization for deployment on edge device.   

## Installation
- for pip installation use command `pip install -r requirements.txt`
- for conda installation use command `conda install --file requirements.txt`
- make sure to run the commands in administrator mode to avoid any issue.
## Model Info
- Achieved an accuracy of 97% on test dataset.
- Model was only trained for 10 epochs.
- Lightweight model with a size of only `18.1 MB` 

## Web Interface
- Used the gradio to make a simple web interface for simplified user interaction.
- Test the app using the sample images.

## Conclusion
- Having any issue or suggestion feel free to reach out.
- Please give it a star if you find it useful.


