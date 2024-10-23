import os
import sys
from ..logger import logging
from ..exception import CustomException
from ..utils import data_preprocessing
from ..utils import TokenPadding, save_object
from ..utils import TokenPaddingTest

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from tensorflow.keras.utils import pad_sequences




class datatransformation():
    def __init__(self):
        self.preprocessor_file_path = os.path.join('archieve', 'preprocessor.pkl')


class MainTransform():
    def __init__(self):
        self.transformation = datatransformation()

            
    def data_transform_initiate(self,train_path, test_text_path, test_target_path):

        try:
            
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_text_path)
            df_labels = pd.read_csv(test_target_path)

            logging.info(f"Reading necessary dataframes.")

            target_Column = 'Category'
            text_column = 'Text'

            input_data_text_features = df_train[[text_column]]
            input_data_target_features = df_train[[target_Column]]

            target_data_text = df_test[[text_column]]
            df_label_data = df_labels[[target_Column]]

            ##Here we apply our first utils function to get data preprocessed and in a list

            logging.info("Preprocessing input text data.")

            input_text_preprocessed = data_preprocessing(df=input_data_text_features,col_name="Text")
            target_data_preprocessed = data_preprocessing(df=target_data_text,col_name='Text')

            ##Now we get them properly tokenized and padded and in numpy format.
            logging.info("Tokenizing and padding the input data.")
            num_words = 5000
            train_tokenizer = TokenPadding()
            X_train, max_length = train_tokenizer.X_train_sequenced(input_text_preprocessed)           
            X_train_padded = np.array(pad_sequences(X_train, maxlen=num_words, padding='pre'))

            test_tokenizer = TokenPaddingTest(train_tokenizer.tokenizer)
            y_train = test_tokenizer.y_token_sequenced(target_data_preprocessed, max_length=max_length)
            y_train_padded = np.array(pad_sequences(y_train, maxlen=num_words, padding='pre'))

            ##Now we will deal with the target data
            logging.info("Tokenization and padding successful.")

            logging.info("Transforming target data.")

            logging.info("Combining training and testing data.")
            target_encoder = LabelEncoder()

            input_data_target_transformed = target_encoder.fit_transform(input_data_target_features.values.ravel())
            df_label_data_transformed = target_encoder.transform(df_label_data.values.ravel())
            
            #print(f'{np.shape(X_train_padded)},{np.shape(input_data_target_transformed)},{np.shape(y_train_padded)},{np.shape(df_label_data_transformed)}')   
            train_arr = np.c_[X_train_padded, np.array(input_data_target_transformed)]
            test_arr = np.c_[y_train_padded, np.array(df_label_data_transformed)]

            logging.info("Saving preprocessing object.")
            save_object(
                file_path = self.transformation.preprocessor_file_path,
                obj= target_encoder
            )
            
            logging.info("Data transformation complete.")
            return train_arr, test_arr, self.transformation.preprocessor_file_path
        
        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise CustomException(e,sys)
        


