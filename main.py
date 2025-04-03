import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Clean text and prepare dataset (same code as in app.py)
    # ...
    return X_padded, y

def train_model(X_train, y_train, X_test, y_test):
    # Build and train the LSTM model (same code as in app.py)
    # ...
    return model

# You can call this function from app.py
def main():
    # Load and preprocess data
    df = load_data('data.csv')
    X, y = preprocess_data(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train, X_test, y_test)
    # Save the model