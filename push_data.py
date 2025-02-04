import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv("MONGO_DB_URL")
print(MONGO_URI)

import certifi
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            
            # Convert DataFrame to JSON string
            json_str = data.to_json(orient="records")

            # Load JSON string into list of dictionaries
            records = json.loads(json_str)
            
            return records

        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database = database
            self.records = records
            self.collection = collection

            self.mongo_client = pymongo.MongoClient(MONGO_URI)
            self.db = self.mongo_client[self.database]

            self.collection = self.db[self.collection]
            self.collection.insert_many(self.records)

            return(len(self.records))

        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    FILE_PATH = os.path.join("Network_Data","phisingData.csv")
    DATABASE = "NetworkAI"
    Collection = "NetworkData"
    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records = networkobj.insert_data_mongodb(records, DATABASE, Collection)
    print(no_of_records)