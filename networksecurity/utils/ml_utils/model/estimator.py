from networksecurity.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

import os, sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkModel:
    def __init__(self,preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            prediction = self.model.predict(x_transform)
            return prediction
        except Exception as e:
            raise NetworkSecurityException(e, sys)