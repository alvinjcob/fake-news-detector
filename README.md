# Fake News Detector 📰

A complete web application to detect whether a news article is Fake or Real. Built using Python, Scikit-learn, and Streamlit.

## Features
- **Machine Learning**: Uses a Multinomial Naive Bayes classification model trained on TF-IDF vectors.
- **Backend API**: Decoupled `predict.py` module handling the ML logic.
- **Interactive UI**: A sleek, user-friendly interface built with Streamlit.
- **Dynamic Feedback**: Warns explicitly with red prompts for Fake news, and confirms with green for Real news.

## Project Structure
```
dapro/
├── dataset/         # Folder containing `Fake.csv` and `True.csv`
├── model/           # Auto-generated folder for saving trained models
├── app.py           # The frontend Streamlit application
├── train.py         # Script to preprocess data, train model, and export artifacts
├── predict.py       # Script containing inference/prediction logic
├── requirements.txt # Project dependencies
└── README.md        # This file
```

## Setup Instructions

1. **Prerequisites**
   Ensure you have Python 3.8+ installed.

2. **Clone / Open the Repository**
   Navigate to the `dapro` project directory in your terminal.

3. **Install Dependencies**
   Run the following command to securely install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Dataset**
   Ensure `Fake.csv` and `True.csv` from the Kaggle dataset are located inside the `dataset` folder.

5. **Train the Model**
   Run the training script to process the datasets and build your Machine Learning models. This will save the `model.pkl` and `vectorizer.pkl` files into the `model/` folder.
   ```bash
   python train.py
   ```
   *This process might take a few moments depending on your system's processing speed.*

6. **Run the Application**
   Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   The application will automatically open in your default browser at `http://localhost:8501`.

## Technical Details
- **Preprocessing**: Texts are lowercased, and we strip punctuation, numbers, and extra whitespaces.
- **Vectorization**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) without specific NLTK stopword dependencies for simplicity, but leveraging Scikit-learn's built-in english stop-word dictionary.
- **Algorithm**: The core model is a probabilistic Multinomial Naive Bayes classifier, proven effective for text classification scenarios.
