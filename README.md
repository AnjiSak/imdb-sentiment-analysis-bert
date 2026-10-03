# IMDB Sentiment Analysis using Machine Learning and BERT

## Overview

This project focuses on classifying IMDB movie reviews as positive or negative using Natural Language Processing (NLP) and Machine Learning.

I explored and compared two different approaches:

1. Traditional Machine Learning using TF-IDF
2. BERT-based sentiment classification using a pre-trained model

The goal was to understand how traditional machine learning models compare with a transformer-based model for a text classification problem.

## Dataset

The project uses the IMDB movie review dataset containing positive and negative movie reviews.

The reviews were processed and divided into training and testing datasets for model development and evaluation.

## Text Preprocessing

For the traditional machine learning models, the text was cleaned before converting it into numerical features.

The preprocessing included basic text cleaning while preserving important negation words such as **"not," "no," and "never"**, since these words can significantly affect the sentiment of a review.

TF-IDF with both unigrams and bigrams was then used to convert the reviews into numerical features.

## Machine Learning Models

I trained and evaluated three traditional machine learning models:

- Logistic Regression
- Linear Support Vector Machine (SVM)
- Random Forest

To compare the models, I evaluated them using:

- Accuracy
- Precision
- Recall
- F1-score

I also recorded training and inference time to understand the computational differences between the models.

## BERT Fine-Tuning

In addition to traditional machine learning models, I used the pre-trained `bert-base-uncased` model for sentiment classification.

Instead of training BERT from scratch, I fine-tuned the pre-trained model for binary sentiment classification using Hugging Face Transformers and PyTorch.

For BERT, I used a balanced subset of:

- 4,000 training reviews — 2,000 positive and 2,000 negative
- 500 testing reviews — 250 positive and 250 negative

A smaller balanced dataset was used because BERT requires significantly more computational resources than the traditional machine learning models.

Unlike the TF-IDF approach, minimal preprocessing was used for BERT because the model relies on contextual information and its tokenizer to understand the relationships between words.

## Model Evaluation

All models were evaluated using the same main classification metrics:

- Accuracy
- Precision
- Recall
- F1-score

The final comparison allows the performance of traditional TF-IDF-based machine learning models to be compared with the fine-tuned BERT model.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Matplotlib
- Jupyter Notebook / Google Colab

## What I Learned

Through this project, I gained hands-on experience with:

- NLP text preprocessing
- TF-IDF feature extraction
- Training and evaluating classification models
- Working with pre-trained transformer models
- BERT tokenization
- Fine-tuning BERT for text classification
- Using Hugging Face Transformers and Trainer
- Comparing traditional machine learning approaches with transformer-based models

## Future Improvements

Some areas I would like to explore further include:

- Hyperparameter tuning
- Training BERT for additional epochs
- Experimenting with other transformer models
- Using a larger dataset for BERT fine-tuning
- Deploying the trained model through an API
