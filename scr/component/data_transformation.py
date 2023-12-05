import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass
from scr.logger import logging
from scr.utils import save_object
from scr.exception import CustomException


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join("artifact", "preprocessing.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["reading score", "writing score"]

            categorical_columns = ["gender","race/ethnicity", "parental level of education","lunch","test preparation course"]
            logging.info("getting preprocessor object")
            num_pipeline = Pipeline(steps = [("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
            logging.info("numerical pipeline successfully")
            cat_pipeline = Pipeline(steps = [("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder()),("scaler",StandardScaler(with_mean = False))])
            logging.info("categorical pipeline successfully")
            preprocessor = ColumnTransformer(transformers = [("num_pipeline",num_pipeline,numerical_columns),("cat_pipeline",cat_pipeline,categorical_columns)])
            logging.info("preprocessor successfully")

            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
    
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("loading data successfull")
            logging.info("getting preprocessor object")
            target_column = "math score"
            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]
            target_feature_test_df = test_df[target_column]
            logging.info("drop and sorting successful")
            preprocessing_obj = self.get_data_transformer_object()
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            ### Concatenate of features array with target array
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info("concatenation successful")
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
                )
            logging.info("Finalization successfully completed")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                )


        except Exception as e:
            raise CustomException(e,sys)
