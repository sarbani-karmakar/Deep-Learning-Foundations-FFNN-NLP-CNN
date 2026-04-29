# Deep Learning Foundations — FFNN, NLP & CNN

A deep learning assignment covering three AI modalities built with TensorFlow/Keras.

## Overview
This project demonstrates foundational deep learning concepts across three domains:
- **Tabular Regression**: Feed-Forward Neural Network (FFNN) on structured customer data
- **Text Classification**: Embedding + GlobalAveragePooling baseline and LSTM model on user feedback
- **Image Classification**: Convolutional Neural Network (CNN) for product image categorization

## What's Covered
- Single neuron forward pass (ReLU vs Sigmoid)
- Data preprocessing, imputation, and StandardScaler without leakage
- FFNN with EarlyStopping for regression (MAE, RMSE)
- Overfitting/underfitting diagnosis from training curves
- Text tokenization, padding, and sequence modeling
- Baseline Embedding model vs LSTM — comparison with transformers
- CNN with Conv2D, MaxPooling, EarlyStopping, and confusion matrix analysis

## Tools & Libraries
Python, TensorFlow 2.20, Keras, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Datasets
- `tabular.csv`: structured customer transaction data (regression target)
- `text.csv`: user feedback text with sentiment/category labels
- `images/`: product images organized by class folders (not included in repo)
