import pandas as pd
import os
import sys
from ..logger import logging
from ..exception import CustomException

#from transformation import MainTransform



class data_ingest():
    def __init__(self):
        self.train_data_path = os.path.join('archieve','train_data.csv')
        self.test_data_path = os.path.join('archieve','test_data.csv')
        self.labels_data_path = os.path.join('archieve','labels.csv')

class IngestCall():
    def __init__(self):
        self.ingestion = data_ingest()
    
    def InitiatinIngestion(self):
        try:
            
            logging.info("Starting data ingestion process.")

            df_train = pd.read_csv(r'notebook/data/df_train')
            df_test = pd.read_csv(r'notebook/data/df_test')
            df_labels = pd.read_csv(r'notebook/data/df_sample')
        
            logging.info("Data files read successfully.")

            df_train = df_train.drop(columns=['ArticleId'], axis=1)
        
            logging.info("'ArticleId' column dropped from training data.")

            df_train.to_csv(self.ingestion.train_data_path, header=True, index=False)
        
            logging.info(f"Training data saved to {self.ingestion.train_data_path}.")

            os.makedirs(os.path.dirname(self.ingestion.train_data_path), exist_ok=True)

            df_test.to_csv(self.ingestion.test_data_path, header=True,index=False)
        
            logging.info(f"Test data saved to {self.ingestion.test_data_path}.")

            df_labels.to_csv(self.ingestion.labels_data_path, header=True, index=False)
        
            logging.info(f"Labels data saved to {self.ingestion.labels_data_path}.")

            return self.ingestion.train_data_path, self.ingestion.test_data_path, self.ingestion.labels_data_path
        
        except Exception as e:
            logging.error("An error occurred during data ingestion.")
            raise CustomException(e,sys)


if __name__ =="__main__":
    logging.info("Data ingestion script started.")

    obj = IngestCall()
    train_data, test_data,labels_data = obj.InitiatinIngestion()
    logging.info("Data ingestion process completed.")
        
#    data_transoformation  = MainTransform()
#    train_arr,test_arr,addditional_info = data_transoformation.data_transform_initiate(train_data1,test_data1,labels_data1)


