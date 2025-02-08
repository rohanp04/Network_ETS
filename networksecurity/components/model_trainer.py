import os
import sys

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.enitity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.enitity.config_entity import ModelTrainerConfig



from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow
from urllib.parse import urlparse

import dagshub
dagshub.init(repo_owner='rohanp04', repo_name='Network_ETS', mlflow=True)


class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, 
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def track_mlflow(self,best_model,classificationmetric):
        with mlflow.start_run():
            f1_score = classificationmetric.f1_score
            precision_score = classificationmetric.precision_score
            recall_score = classificationmetric.recall_score

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision_score", precision_score)
            mlflow.log_metric("recall_score", recall_score)
            mlflow.sklearn.log_model(best_model,"model")
        
    def train_model(self, x_train, y_train, x_test, y_test):
        models = {
            "LogisticRegression": LogisticRegression(verbose=1),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(verbose=1),
            "RandomForestClassifier": RandomForestClassifier(verbose=1),
        }
        params = {
            "DecisionTreeClassifier": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "RandomForestClassifier":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "GradientBoostingClassifier":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "LogisticRegression":{},
            "AdaBoostClassifier":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            } 
        }
        model_report: dict = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, 
                                             models=models, params=params)
        
        best_model_name = max(model_report, key=model_report.get)  # Get the best model's name
        best_model = models[best_model_name]  # Retrieve the best model


        y_train_predict = best_model.predict(x_train)

        classification_metric_train = get_classification_score(y_true=y_train, y_pred=y_train_predict)

        ##Track the MLFlow
        self.track_mlflow(best_model,classification_metric_train)

        y_test_pred = best_model.predict(x_test)
        classification_metric_test = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Network_Model = NetworkModel(preprocessor= preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=NetworkModel)

        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_metric_train,
                             test_metric_artifact=classification_metric_test
                            )
        logging.info(f"model trainer artifact data is as follows: {model_trainer_artifact}")
        return model_trainer_artifact
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            model = self.train_model(x_train,y_train,x_test,y_test)

        except Exception as e:
            raise NetworkSecurityException(e,sys)