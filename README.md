Twitter Sentiment Analysis
A sentiment analysis project for classifying tweets as positive or negative using a GRU-based neural network.
Overview

Dataset: 1.6M tweets (cleaned as clean_data.csv).
Model: Keras Sequential with Embedding, GRU, and Dense layers.
Files:
EDA.ipynb: Data cleaning.
preprocessing.py: Text cleaning and tokenization.
train.ipynb: Model training and saving.
app.ipynb: Flask app for predictions.


Output: sentiment_model.h5, tokenizer.pkl.

Requirements

Python 3.12+
Libraries: pandas, scikit-learn, tensorflow, flask

pip install pandas scikit-learn tensorflow flask

Usage

Run EDA.ipynb to clean data.
Run train.ipynb to train and save the model.
Run app.ipynb and visit http://127.0.0.1:5000 to analyze tweets.

License
MIT License
