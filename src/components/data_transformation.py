import sys
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import   OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from utils import save_object

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformataion:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            num_cols = ['writing_score','reading_score']
            cat_cols = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())

                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder())

                ]
            )
            logging.info(f'Categorical Columns: {cat_cols}')
            logging.info(f'Numerical Columns: {num_cols}')
            preprocessor = ColumnTransformer(
                [
                    ('num_pipline',num_pipeline,num_cols),
                    ('cat_pipline',cat_pipeline,cat_cols)
            
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformer_obj()
            tgt_col_name = 'math_score'
            num_cols = ['writing_score','reading_score']
            input_feature_train_df = train_df.drop(columns = [tgt_col_name],axis = 1)
            tgt_feature_train_df = train_df[tgt_col_name]
            input_feature_test_df = test_df.drop(columns = [tgt_col_name],axis = 1)
            tgt_feature_test_df = test_df[tgt_col_name]
            logging.info(
                f'Applying preprocessing object on training dataframe and testing dataframe.'
            )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(tgt_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(tgt_feature_test_df)]

            logging.info(f'Saved preprocessing object.')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,

            )
        except Exception as e:
            raise CustomException(e,sys)
        
