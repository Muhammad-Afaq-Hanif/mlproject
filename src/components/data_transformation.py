import sys
import os 
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exceptional import log_exception
from src.logger import logging
from src.utils import save_object

@dataclass
class TransformFilePath:
   transform_data_path:str = os.path.join('artifacts','transform.pkl')

class transformationDetail:
    def __init__(self):
        self.transformpath = TransformFilePath()
        
    def transformation(self):
        try:
            numerical_data = ['reading_score','writing_score']
            categorical_data = [
                'gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course'
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('standardScalar',StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoding',OneHotEncoder()),
                    ('StandardScaler',StandardScaler(with_mean=False))
                ]
            )
            
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_data),
                    ('cat_pipline',cat_pipeline,categorical_data)
                ]
            )
            
            logging.info("Transformation process is working.")
            return preprocessor
            
            
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info
            log_exception(exc_type, exc_value, exc_traceback)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Test and Train data is readed.")
            preprocessor_obj = self.transformation()
            
            target_column = 'math_score' 
            train_dataset = train_data.drop(columns=target_column)
            train_dataset_target = train_data[target_column ]
            test_dataset = test_data.drop(columns=target_column )
            test_dataset_target = test_data[target_column ]
            
            features_train_arr = preprocessor_obj.fit_transform(train_dataset)
            features_test_arr = preprocessor_obj.transform(test_dataset)
            logging.info("Transformation is done.")
            
            train_arr = np.c_[features_train_arr,np.array(train_dataset_target)]
            test_arr = np.c_[features_test_arr,np.array(test_dataset_target)]
            
            save_object(
                file_path = self.transformpath.transform_data_path,
                object = preprocessor_obj
            )
            logging.info("Pickle File is created.")
            
            return (
                train_arr,
                test_arr,
                self.transformpath.transform_data_path
            )
            
        except Exception:
            exc_typ, exc_value, exc_traceback = sys.exc_info()
            log_exception(exc_typ, exc_value, exc_traceback)