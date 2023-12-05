import os
import sys
from scr.logger import logging
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.ensemble import (AdaBoostRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from scr.exception import CustomException
from dataclasses import dataclass
from scr.utils import save_object,evalute_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "models.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("start of the training process")
            x_train,y_train,x_test,y_test = (train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])
            ## model dictionary
            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor()
            }
            model_report:dict = evalute_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("no best model found",sys)
                logging.info("we cannot find the best model")
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"r2 square is {r2_square}")
            logging.info(f"best model is {best_model_name}")
            logging.info(f"best model score is {best_model_score}")
            logging.info(f"All the model_report with training score {model_report}")
            logging.info("training process is completed")
            
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
            