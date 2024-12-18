
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd


from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainConfig





def main():
    logging.info("STARTING DATA INGESTION PIPELINE...")
    
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    logging.info("DATA TRANSFORMATION")
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    logging.info("MODEL TRAINING")
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))





if __name__ == '__main__':
    main()