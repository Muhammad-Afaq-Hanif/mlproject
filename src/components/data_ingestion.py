import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exceptional import log_exception
from src.logger import logging
from src.components.data_transformation import transformationDetail

@dataclass
class data_path: #dataingestionconfig
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','raw.csv')
    
class DataIngestion:
    def __init__(self):
        self.path = data_path() #Ingestionconfig
    
    def initiateDataIngestion(self):
        logging.info("Entered ingestion method or component")
        try:
            df = pd.read_csv(r"notebook\data\stud.csv")
            logging.info("Raw data is read.")
            os.makedirs(os.path.dirname(self.path.raw_data_path),exist_ok=True)
            logging.info("Directory is Made.")
            df.to_csv(self.path.raw_data_path,header=True,index=False)
            logging.info("Data is stored.")
            train_data, test_data = train_test_split(df,test_size=0.2,random_state=42)
            train_data.to_csv(self.path.train_data_path,index=False,header=True)
            test_data.to_csv(self.path.test_data_path,header=True,index=False)
            logging.info("Data is separted and stored in the files")
            return(
                self.path.train_data_path,
                self.path.test_data_path
            )
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            log_exception(exc_type,exc_value,exc_traceback)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiateDataIngestion()
    transformation = transformationDetail()
    transformation.initiate_data_transformation(train_data, test_data)
    

