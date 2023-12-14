from flask import Flask, render_template, request
import os
import torch
import speech_recognition as sr
from transformers import AutoTokenizer
from model import SentimentClassifier, load_model_and_tokenizer, predict_sentiment

app = Flask(__name__)

# Load saved model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r'E:\ngocquy_python\AI\bully\bully_detection_model.pth' 
tokenizer_path = r'E:\ngocquy_python\AI\bully\bully_detection_model'   
loaded_model, loaded_tokenizer = load_model_and_tokenizer(SentimentClassifier, AutoTokenizer, model_path, tokenizer_path, device)

class_names = {0: 'Non-Violence', 1: 'Violence'}  # Update with your actual class names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sentence = request.form['sentence']
        predicted_label = predict_sentiment(sentence, loaded_tokenizer, loaded_model, device)

        result = class_names[predicted_label]
        return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
