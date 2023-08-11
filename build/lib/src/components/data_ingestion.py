import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import pymongo # this was missing 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation


## Intitialize the Data Ingetion Configuration


@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

## create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            df=pd.read_csv(os.path.join('notebooks','insurance.csv'))
            logging.info('Dataset read from pandas Dataframe')
            client = pymongo.MongoClient("mongodb+srv://smitha:ineuron@cluster0.pge2ml4.mongodb.net/")
            db = client['insuranceFraud']
            collection = db['insuranceTable']

            data_dict = df.to_dict('records') # here what does data show thre was not dataframe as data it should be df
            collection.insert_many(data_dict)
            logging.info("data is inserted")
            data = list(collection.find())
            insurance_dataframe = pd.DataFrame(data) # insurance_dataframe was not used and insted df was used in the project.
            client.close()
            logging.info('Dataset read from pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            insurance_dataframe.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test split')
            train_set,test_set=train_test_split(insurance_dataframe,test_size=0.4,random_state=0)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
  
            
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)
