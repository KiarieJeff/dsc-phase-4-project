from flask import Flask, request, jsonify
import pickle
from nltk.tokenize import word_tokenize,TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re


app = Flask(__name__)

# Load the trained model
with open('model_rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def clean_and_preprocess_text(text):
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)
    # Convert tokens to lowercase
    tokens = [token.lower() for token in tokens]
    # Remove mentions (words starting with '@') and URLs
    tokens = [token for token in tokens if not token.startswith('@') and not token.startswith('http')]
    # Remove punctuation and numbers using regular expressions
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Apply stemming using the Porter Stemmer
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    cleaned_text = ' '.join(stemmed_tokens) 
    return cleaned_text

def map_to_new_column(value):
    if value == 0.0:
        return 'negative'
    elif value == 1.0:
        return 'positive'
    elif value == 2.0:
        return 'neutral'
    else:
        return 'unknown'  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        x = data['x']  # Assuming you send a JSON request with 'x' as input
        processed_text = clean_and_preprocess_text(x)
        prediction = model.predict([processed_text])
        prediction = map_to_new_column(prediction[0])
        response = {'prediction': prediction}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True)