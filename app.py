from flask import Flask, request, jsonify
import pickle
from nltk.tokenize import word_tokenize,TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import numpy as np
from scipy.sparse import csr_matrix


app = Flask(__name__)

# Load the trained model
with open('sent_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vect_file:
    vectorizer = pickle.load(vect_file)    

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

def sentiment_predict(text):
    cl_txt = clean_and_preprocess_text(text)
    tfidf_vector_single = vectorizer.transform([cl_txt])
    csr_mat = csr_matrix(tfidf_vector_single)
    csr_array = csr_mat.toarray()
    pred = model.predict(csr_array)
    rounded_arr = np.round(pred)

    # Extract the rounded values 
    a, b, c = rounded_arr[0]

    # Determine the sentiment label based on the rounded values
    if a == 1:
        return "Negative"
    elif b == 1:
        return "Positive"
    elif c == 1:
        return "Neutral"
    else:
        return "Unknown"
  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        x = data['x']  # Assuming you send a JSON request with 'x' as input
        prediction = sentiment_predict(x)
        response = {'prediction': prediction}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True)