from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load dataset
file_path = 'spam.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError("Dataset file not found! Please ensure 'spam.csv' is in the project directory.")

df = pd.read_csv(file_path, encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

df['processed_text'] = df['text'].apply(preprocess_text)

# Convert text to numerical data
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['processed_text']).toarray()
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Evaluate model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Default route
@app.route('/')
def home():
    return "Welcome to the Spam Detection API! Use the '/predict' endpoint to check messages."

# API endpoint for spam detection
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text]).toarray()
    prediction = svm_model.predict(text_vector)[0]
    return jsonify({'prediction': 'Spam' if prediction == 1 else 'Ham'})

if __name__ == '__main__':
    app.run(debug=True)