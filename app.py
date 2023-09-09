import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import nltk
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
# Load the trained model
# Replace with your model filename
model = load_model('Model_LSTM.h5')

# Preprocess functions

tokenizer = Tokenizer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


stop_words = stopwords.words('english')
more_stopwords = ['u', 'im', 'c']
stop_words = stop_words + more_stopwords


def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text


stemmer = SnowballStemmer("english")


def stemm_text(text):
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text


d = {0: 'JOY', 1: 'NEUTRAL', 2: 'OPTIMISM', 3: 'UPSET'}

# Streamlit app
st.title('SENTIMENTAL ANALYSIS')
max_seq_length = 250
# Input text box
input_text = st.text_area('Enter your text:', '')

# Preprocess and predict
if st.button('Predict'):
    if input_text:
        # Preprocess the input text
        cleaned_text = clean_text(input_text)
        text_without_stopwords = remove_stopwords(cleaned_text)
        stemmed_text = stemm_text(text_without_stopwords)

        # Tokenize and encode the text
        input_sequence = tokenizer.texts_to_sequences([stemmed_text])

        # Pad sequences
        input_sequence = pad_sequences(input_sequence, maxlen=max_seq_length)

        # Make predictions
        predicted_probabilities = model.predict(input_sequence)

        # Convert predictions to class label
        p = np.argmax(predicted_probabilities)

        st.write(f'Predicted Class: {d[p]}')
    else:
        st.write('Please enter some text for prediction.')
