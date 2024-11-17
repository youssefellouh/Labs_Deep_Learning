# Labs Deep Learning

## Project Overview
This repository contains various deep learning projects focusing on Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN). Each subdirectory is dedicated to a specific project, including Jupyter notebooks and code implementations for different tasks such as age detection, fruit and vegetable classification, and stock price prediction.

### Directory Structure
```
├── Labs_CNN
│   ├── AgeDetect CNN
│   │   └── detection_age.ipynb  # Notebook for age detection using a CNN model
│   └── Fruits_Vegetables
│       ├── app.py               # Python script for deploying the fruit and vegetable classification model
│       └── ML.ipynb             # Notebook for training and evaluating the fruit and vegetable classifier
├── Labs_RNN
│   ├── code_RNN.ipynb           # Notebook with RNN implementation for time series data
│   ├── data                     # Data directory for stock price analysis
│   │   ├── Google_Stock_Price_Test.xls
│   │   └── Google_Stock_Price_Train.xls
│   ├── data2                    # Data directory for apple stock price analysis
│   │   ├── apple_prices.csv
│   │   └── normalized_apple_prices.csv
│   ├── data3                    # Text data directory for RNN-based text analysis
│   │   ├── Antigon.txt
│   │   ├── La_Boîte_à_merveilles.txt
│   │   └── Le_Dernier_Jour_d'un_Condamné.txt
│   ├── images                   # Directory with image assets for visualization
│   │   ├── image1.png
│   │   ├── image2.png
│   │   ├── image3.png
│   │   ├── image4.png
│   │   ├── image5.png
│   │   └── image.png
│   ├── RNN1.ipynb               # Notebook for advanced RNN modeling
│   └── Simple_RNN.ipynb         # Notebook for a basic RNN implementation
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies for the project
```

## Project Descriptions

### 1. Labs_CNN
- **AgeDetect CNN**: A convolutional neural network model that predicts age based on input data. The `detection_age.ipynb` notebook contains the code for data preprocessing, model architecture, training, and evaluation.
- **Fruits_Vegetables**: This project includes a model for classifying images of fruits and vegetables. The `ML.ipynb` notebook provides the model development process, and `app.py` is used for deploying the model as an application.

### 2. Labs_RNN
- **code_RNN.ipynb**: This notebook demonstrates RNN implementation for stock price prediction, using datasets located in the `data` and `data2` directories.
- **Simple_RNN.ipynb** and **RNN1.ipynb**: These notebooks cover the basics and advanced aspects of RNNs for various time series and text data analyses.
- **data**, **data2**, and **data3** directories contain relevant datasets for training and testing models.
## Installation
1- Clone the repository:
```
https://github.com/youssefellouh/Labs_Deep_Learning.git
cd Labs_Deep_Learning
```
2- reate and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```
3-Install the required packages:
```
pip install -r requirements.txt
```

## Usage
Open any of the Jupyter notebooks in your preferred environment (e.g., Jupyter Notebook, JupyterLab, or VSCode) and follow the instructions in the cells to execute the code and reproduce the results.



