from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.enitity.config_entity import DataIngestionConfig
from networksecurity.enitity.config_entity import TrainingPipelineConfig

import sys

if __name__ == "__main__":
    try:
        Trainingpipelineconfig = TrainingPipelineConfig()
        Dataingestionconfig = DataIngestionConfig(Trainingpipelineconfig)
        data_ingestion = DataIngestion(Dataingestionconfig)
        logging.info("Starting the Data Ingestion")
        dataingestionartifact = data_ingestion.initate_data_ingestion()
        print(dataingestionartifact)

    except Exception as e:
        raise NetworkSecurityException(e,sys)