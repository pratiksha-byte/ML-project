import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object,evaluate_models

@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split the data into training and testing")
            X_train,Y_train,X_test,Y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models =LogisticRegression()

            models.fit(X_train,Y_train)
            logging.info("model training completed")
            
            save_object(
                self.model_trainer_config.trained_model_file_path,
                obj=models
            )
            logging.info('saving model complete')


            predicted = models.predict(X_test)

            accuracy = accuracy_score(Y_test,predicted)

            return accuracy
        
        except Exception as e:
            raise CustomException(e,  sys)

