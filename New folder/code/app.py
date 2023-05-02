import os
import nltk
nltk.download('all')

from flask import Flask, render_template, request
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model

loaded_model = load_model('spam.pkl')
cv = pickle.load(open('cv1.pk1','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/spam', methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        message = request.form['message']
        data = message

        new_review = str(data)
        print(new_review)
        new_review = re.sub('[^a-zA-Z]', ' ', new_review)
        new_review = new_review.lower()
        new_review = new_review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('hot')
        new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
        new_review = ' '.join(new_review)
        new_corpus = [new_review]
        new_X_test = cv.transform(new_corpus).toarray()
        print(new_X_test)

        prediction = loaded_model.predict(new_X_test)

        if prediction[0][0] == 1:
            return render_template('result.html', prediction="spam")
        else:
            return render_template('result.html', prediction="Not a Spam")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, port=port)
