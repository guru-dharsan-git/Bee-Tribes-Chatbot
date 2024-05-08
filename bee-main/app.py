from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import random
import json
app = Flask(__name__)

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Load the intents file
with open('intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

def get_response(message):
    # Tokenize the input message
    inputs = tokenizer(message, return_tensors='pt', max_length=128, truncation=True)

    # Get the model's prediction
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()

    # Return a random response from the intent based on the predicted class
    intent_responses = data['intents'][predicted_class]['responses']
    return random.choice(intent_responses)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        request_data = request.json
        user_input = request_data.get('user_input')

        if user_input is None:
            raise ValueError('Missing "user_input" field in the request JSON.')

        response = get_response(user_input)
        return jsonify({'response': response})

    except Exception as e:
        print(e)
        return jsonify({'error': 'Invalid input'})

if __name__ == '__main__':
    app.run(debug=True)
