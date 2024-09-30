# Sentiment Analysis Project

## Problem Statement

The objective of this project is to create a binary polarity classifier that can distinguish between positive and negative sentiments using a dataset containing movie reviews. The dataset comprises 5,331 positive and 5,331 negative sentences, sourced from [Cornell's Movie Review Data](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz). 

### Dataset Preparation

The dataset is split into three parts:
- **Training Set**: The first 4,000 positive and the first 4,000 negative reviews.
- **Validation Set**: The next 500 positive and the next 500 negative reviews.
- **Test Set**: The final 831 positive and the final 831 negative reviews.

### Data Description

The dataset contains two files:
- `rt-polarity.pos`: Contains positive reviews.
- `rt-polarity.neg`: Contains negative reviews.

### Implementation Approaches

This project implements multiple machine learning (ML) and deep learning (DL) approaches to classify the sentiments, including:

1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Random Forest Classifier**
4. **Long Short-Term Memory (LSTM) Neural Network**
5. **BERT (Bidirectional Encoder Representations from Transformers)**

### Code Overview

#### 1. Data Download and Preparation

The dataset is downloaded and extracted, followed by preprocessing steps including tokenization, removing special characters, and lemmatization. 

#### 2. Feature Extraction

Various feature extraction techniques such as TF-IDF and GloVe embeddings are used to convert text data into numerical format suitable for model training.

#### 3. Model Training

The models are trained on the training set, validated using the validation set, and evaluated on the test set. 

#### 4. Evaluation Metrics

The performance of the models is evaluated using metrics such as accuracy, precision, recall, and F1-score.

### Results

A summary of the results for each model will be provided after training and evaluation. 

### Requirements

To run this project, you'll need the following Python packages:
- numpy
- pandas
- re
- nltk
- scikit-learn
- tensorflow (for LSTM)
- torch (for BERT)

### How to Run

1. Clone this repository.
2. Ensure all required packages are installed.
3. Run the Jupyter notebooks or Python scripts provided in this project.

### References

- [Cornell Movie Review Data](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz)
- [Cornell Movie Review Data README](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.README.1.0.txt)

