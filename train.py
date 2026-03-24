import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

def preprocess_text(text):
    """
    Cleans textual data by making it lowercase, removing
    punctuation, numbers, and extra spaces.
    """
    # Convert to lowercase
    text = str(text).lower()
    # Remove punctuation and non-word characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = text.strip()
    return text

print("Loading datasets...")
try:
    fake_df = pd.read_csv('dataset/Fake.csv')
    true_df = pd.read_csv('dataset/True.csv')
except FileNotFoundError:
    print("Error: Datasets not found in 'dataset/' directory.")
    print("Please make sure 'Fake.csv' and 'True.csv' are present in the 'dataset' folder.")
    exit()

# Add labels
# 0 for FAKE
fake_df['label'] = 0 
# 1 for REAL
true_df['label'] = 1 

# Concatenate datasets
print("Concatenating datasets...")
df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

# Combine title and text to create a more comprehensive feature
df['text'] = df['title'] + " " + df['text']

# Drop missing values if any
df = df.dropna(subset=['text'])

print("Preprocessing text...")
df['text'] = df['text'].apply(preprocess_text)

# Features and target
X = df['text']
y = df['label']

# Split data into training and testing sets (80% train, 20% test)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization using TF-IDF
print("Vectorizing text (TF-IDF)...")
# We ignore English stop-words automatically using the builtin list
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Multinomial Naive Bayes model
print("Training Multinomial Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
print("Evaluating model...")
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

# Save model and vectorizer with Pickle for later use in predict.py and Streamlit app
print("Saving model and vectorizer...")
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Training finished! Model and Vectorizer have been successfully saved in 'model/' directory.")
