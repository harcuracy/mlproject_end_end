import os
import sys
from scr.logger import logging
import pandas as pd
import dill
import numpy as np

from sklearn.metrics import r2_score
from scr.exception import CustomException
from sklearn.model_selection import train_test_split


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path) 
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def split_dataset(dataset, test_size=None,random_state=None):
    try:
        train, test = train_test_split(dataset, test_size=test_size, random_state=random_state)
        return (train, test)
    except Exception as e:
        raise CustomException(e, sys)


def evalute_models(x_train,y_train,x_test,y_test,models):
    report = {}
    try:
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train,y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_model_score = r2_score(y_train_pred,y_train)
            test_model_score = r2_score(y_test_pred,y_test)
            report[list(models.keys())[i]] = test_model_score
        return report
        
    except Exception as e:
        raise CustomException(e, sys)

        



def make_directory(dirName):
    try:
        if not os.path.exists(dirName) or os.path.getsize(dirName)==0:
            dirName = os.path.dirname(dirName)
            os.makedirs(dirName, exist_ok=True)
    except Exception as e:
        raise CustomException(e, sys)
