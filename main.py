from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.components.data_validation import DataValidation
from networksecurity.logging.logger import logging
from networksecurity.enitity.config_entity import DataIngestionConfig, DataValidationConfig
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

    except Exception as e:
        raise NetworkSecurityException(e,sys)