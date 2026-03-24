import pickle
import re
import os

# Paths to the saved model and vectorizer
MODEL_PATH = 'model/model.pkl'
VECTORIZER_PATH = 'model/vectorizer.pkl'

def load_objects():
    """
    Loads and returns the trained machine learning model and vectorizer.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Model or Vectorizer not found. Please run train.py first to generate the models.")
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

def preprocess_text(text):
    """
    Cleans textual data identical to the preprocessing done during training.
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

def predict_news(news_text):
    """
    Predicts if the given news text is FAKE or REAL.
    Returns the classification label and confidence score.
    """
    model, vectorizer = load_objects()
    
    # Preprocess text input
    cleaned_text = preprocess_text(news_text)
    
    # Convert text to TF-IDF features
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]
    
    # Map back to labels (0: FAKE, 1: REAL based on train.py)
    if prediction == 0:
        label = "FAKE"
        confidence = probabilities[0]
    else:
        label = "REAL"
        confidence = probabilities[1]
        
    return label, confidence

if __name__ == "__main__":
    # Test script locally if executed directly
    sample_text = "Scientists have discovered a new planet made entirely of diamonds!"
    try:
        label, conf = predict_news(sample_text)
        print(f"Sample prediction for test sentence: {label} (Confidence: {conf:.2f})")
    except Exception as e:
        print(f"Error: {e}")
