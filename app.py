import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Download NLTK stopwords
nltk.download('stopwords')

# Load dataset function (using a sample dataset from IMDB for example)
@st.cache
def load_data():
    # Replace with your dataset path or URL if necessary
    df = pd.read_csv('movies-dataset-mzd4mydvila.csv')  # Adjust the path as needed
    return df

df = load_data()

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Clean reviews
df['cleaned_review'] = df['review'].apply(clean_text)

# Tokenization & Padding
max_words = 5000
max_length = 200
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['cleaned_review'])
X_sequences = tokenizer.texts_to_sequences(df['cleaned_review'])
X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post', truncating='post')

# Convert sentiment labels to binary
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})
y = df['sentiment'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Build the LSTM model
embedding_dim = 64
model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_length),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the LSTM model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the LSTM model
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Display model evaluation results in Streamlit
st.write(f"### Model Evaluation Results:")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
st.write(f"Precision: {precision_score(y_test, y_pred):.4f}")
st.write(f"Recall: {recall_score(y_test, y_pred):.4f}")
st.write(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# Classification report
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion matrix visualization
st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - LSTM')
st.pyplot(fig)

# Show a wordcloud of frequently used words
from wordcloud import WordCloud
all_reviews = ' '.join(df['cleaned_review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)

st.write("### Wordcloud of Frequent Words in Reviews")
st.image(wordcloud.to_array())

# Allow users to input their own review for prediction
st.write("### Sentiment Prediction for Custom Review")
user_review = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if user_review:
        cleaned_review = clean_text(user_review)
        user_sequence = tokenizer.texts_to_sequences([cleaned_review])
        user_padded = pad_sequences(user_sequence, maxlen=max_length, padding='post', truncating='post')
        user_prediction = model.predict(user_padded)
        sentiment = 'Positive' if user_prediction > 0.5 else 'Negative'
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter a review to predict sentiment.")
