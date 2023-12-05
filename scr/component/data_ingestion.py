import os
import sys
import pandas as pd
from scr.exception import CustomException
from scr.logger import logging
from scr.utils import split_dataset
from scr.utils import make_directory
from dataclasses import dataclass
from scr.component.model_trainer import ModelTrainer
from scr.component.data_transformation import DataTransformation
from scr.component.data_transformation import DataTransformationConfig



@dataclass

class DataIngestionConfig:

    train_data_path:str = os.path.join("artifact", "train.csv")

    test_data_path:str = os.path.join("artifact", "test.csv")

    raw_data_path:str = os.path.join("artifact", "data.csv")

class DataIngestion:

    def __init__(self):

        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Starting data ingestion")

        try:

            df = pd.read_csv(r"data\studs_P.csv")

            logging.info("Reading my dataset locally")

            make_directory(self.ingestion_config.train_data_path)

            logging.info("making directory successful")

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("saving main dataset in directory successfully")

            train_set,test_set = split_dataset(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) 

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("ingestion of the data is successfully")

            return(

                self.ingestion_config.train_data_path,

                self.ingestion_config.test_data_path
                )

        except Exception as e:

            raise CustomException(e,sys)

            logging.info("ingestion of the data is unsuccessful")

if __name__ == "__main__":

    obj=DataIngestion()

    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    train_array,test_array,preprocess_path = data_transformation_obj.initiate_data_transformation(train_data,test_data)
    model_obj = ModelTrainer()
    score = model_obj.initiate_model_trainer(train_array,test_array)
    print(score)
    