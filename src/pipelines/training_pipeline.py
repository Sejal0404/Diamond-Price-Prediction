import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        logging.info("Starting training pipeline execution")
        
        # Data Ingestion
        logging.info("Initiating data ingestion")
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        logging.info(f"Data ingestion completed. Train path: {train_data_path}, Test path: {test_data_path}")

        # Data Transformation and save preprocessor
        logging.info("Initiating data transformation")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info(f"Data transformation completed. Preprocessor saved at: {preprocessor_path}")

        # Model Training
        logging.info("Initiating model training")
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)
        logging.info("Model training completed successfully")

    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}", exc_info=True)
        raise CustomException(e, sys)
