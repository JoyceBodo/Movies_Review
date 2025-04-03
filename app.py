import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

nltk.download('stopwords')

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/JoyceBodo/Movies_Review/main/movies-dataset.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Preprocessing function
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        return text
    return ""

df['cleaned_review'] = df['review'].apply(clean_text)

def plot_wordcloud():
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['cleaned_review']))
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Reviews")
    st.pyplot(plt)

# TF-IDF and ML Models
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_predictions = nb_model.predict(X_test_tfidf)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_predictions = lr_model.predict(X_test_tfidf)

# LSTM Model Preparation
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['cleaned_review'])
X_sequences = tokenizer.texts_to_sequences(df['cleaned_review'])
X_padded = pad_sequences(X_sequences, maxlen=200, padding='post', truncating='post')

X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_padded, df['sentiment'], test_size=0.2, random_state=42)

model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=200),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Streamlit UI
st.title("Movie Reviews Sentiment Analysis")

if st.button("Show Word Cloud"):
    plot_wordcloud()

st.subheader("Model Evaluation")
st.write("Naïve Bayes Accuracy:", accuracy_score(y_test, nb_predictions))
st.write("Logistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))

user_review = st.text_area("Enter your review:")
if st.button("Predict Sentiment"):
    cleaned_review = clean_text(user_review)
    user_sequence = tokenizer.texts_to_sequences([cleaned_review])
    user_padded = pad_sequences(user_sequence, maxlen=200, padding='post', truncating='post')
    user_prediction = model.predict(user_padded)
    sentiment = 'Positive' if user_prediction > 0.5 else 'Negative'
    st.write(f"Predicted Sentiment: {sentiment}")