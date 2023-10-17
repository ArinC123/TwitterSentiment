
import pandas as pd
import re
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('C:/Users/Admin/Desktop/twittersentiment/static/twitter_training.csv', header=None, names=['ID', 'Product', 'Sentiment', 'Tweet'])
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = word_tokenize(text)
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        return ' '.join(words)
    else:
        return ''

data['Tweet'] = data['Tweet'].apply(preprocess_text)

# TF-IDF vectorization and model training
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(data['Tweet'])
y = data['Sentiment']
model = MultinomialNB()
model.fit(X_tfidf, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.form['user_input']
    user_input = preprocess_text(user_input)
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    prediction = model.predict(user_input_tfidf)
    return render_template('index.html', sentiment="Sentiment: " + prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
