from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.logging.logger import logging
from networksecurity.enitity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from networksecurity.enitity.config_entity import TrainingPipelineConfig

import sys

if __name__ == "__main__":
    try:
        Training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(Training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Starting the Data Ingestion")
        data_ingestion_artifact = data_ingestion.initate_data_ingestion()
        logging.info("Data Ingestion completed")
        print(data_ingestion_artifact)
        data_validation_config = DataValidationConfig(Training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Starting the Data Validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation completed")
        print(data_validation_artifact)
        data_transformation_config=DataTransformationConfig(Training_pipeline_config)
        logging.info("data Transformation started")
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("data Transformation completed")

        logging.info("Model Training sstared")
        model_trainer_config=ModelTrainerConfig(Training_pipeline_config)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()

        logging.info("Model Training artifact created")

    except Exception as e:
        raise NetworkSecurityException(e,sys)