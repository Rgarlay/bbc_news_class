import os
import sys

import pandas as pd
import numpy as np
import dill
from src.news_project.exception import CustomException
import nltk 
from nltk.corpus import stopwords
import re
from tensorflow.keras.preprocessing.text import Tokenizer

from nltk.stem import WordNetLemmatizer

#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('punkt')
#nltk.download('stopwords')


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
            raise CustomException(e,sys)
    
def load_object(file_path):

    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
                
    except Exception as e:
        raise CustomException(e,sys)

def data_preprocessing(df,col_name):
    try:
        lemma = WordNetLemmatizer()       
        combined_stopwords = set(stopwords.words('english'))
        text_cleaned = []
        for i in range(len(df)):
            # Clean the text
            text = re.sub('[^a-zA-Z0-9]', ' ', df[col_name][i])
            text = text.lower()
            text = text.split()
            text = [word for word in text if word not in combined_stopwords]
            text = [lemma.lemmatize(word) for word in text]
            cleaned_text = ' '.join(text)
            text_cleaned.append(cleaned_text)
            
    except Exception as e:
        raise CustomException(e,sys)

    return text_cleaned


class TokenPadding:
    def __init__(self):
        self.num_words = 1000  # Default number of words
        self.tokenizer = Tokenizer(num_words=self.num_words, oov_token='<OOV>')


    def X_train_sequenced(self, sequence, num_words=1000):
        try:
            self.tokenizer.fit_on_texts(sequence)

            max_length = max(len(seq) for seq in sequence)

            cleaned_data = []
            for i in range(len(sequence)):
                sequences_encoded = self.tokenizer.texts_to_sequences([sequence[i]])
                cleaned_data.append(sequences_encoded[0])
            
        except Exception as e:
            raise CustomException(e, sys)
        
        return cleaned_data, max_length
    

class TokenPaddingTest(TokenPadding):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer  

    def y_token_sequenced(self, sequence, max_length=None):
        try:

            cleaned_data = []
            for i in range(len(sequence)):
                sequences_encoded = self.tokenizer.texts_to_sequences([sequence[i]])
                cleaned_data.append(sequences_encoded[0])

        except Exception as e:
            raise CustomException(e, sys)
        
        return cleaned_data



