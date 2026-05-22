# System imports
import sys
import os 
import pandas as pd
# Imports for MLflow
import argparse
import mlflow
import mlflow.sklearn
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset

# Silence warnings for cleaner logs
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
# Silence MLflow specific warnings
logging.getLogger("mlflow").setLevel(logging.ERROR)
# Silence the specific Deprecation and Security warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pickle or cloudpickle format.*")

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import pipeline components
from src.Pipeline.Data_Ingestion import DataIngestion, DataIngestionConfig
from src.Pipeline.Data_Transformation import Preprocess, PreprocessConfig
from src.Pipeline.Model_Trainer import ModelTrainer, ModelTrainerConfig
from src.Pipeline.Model_Evaluation import evaluate_model

# Custom Load object and Logger and Exceptions
from src.utils.file_logs import get_logger
from src.utils.exceptions import CustomException


log = get_logger()

def main(data_path, experiment_name):
    # Set Mlflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        try:
            log.info(f"  ------ Starting Pipeline for Experiment: {experiment_name} ------")

            # Data Ingestion
            log.info(" ---------- Step 1: Starting Data Ingestion ----------")
            ingestion_config = DataIngestionConfig()
            data_ingestion = DataIngestion(config=ingestion_config)
            df = data_ingestion.load_data(file_path=data_path)

            # Log the original data path as a parameter and artifact to MLflow
            mlflow.log_param("original_data_path", data_path)
            mlflow.log_artifact(data_path, artifact_path="data/original")
            log.info(" ---------- Step 1: Data Ingestion completed successfully ----------")

            # Data Transformation
            log.info(" ---------- Step 2: Starting Data Transformation ----------")
            preprocess_config = PreprocessConfig()
            preprocessor = Preprocess(config=preprocess_config)
            preprocessor.process_data(df=df)

            # Log the paths of the train, test, and validation splits as parameters and artifacts to MLflow
            mlflow.log_artifact(preprocess_config.train_path, artifact_path="data/splits")
            mlflow.log_artifact(preprocess_config.test_path, artifact_path="data/splits")
            mlflow.log_artifact(preprocess_config.app_validation_path, artifact_path="data/validation")

            # Load the datasets into MLflow
            train_dataset: PandasDataset = mlflow.data.from_pandas(pd.read_csv(preprocess_config.train_path), source=preprocess_config.train_path)
            test_dataset: PandasDataset = mlflow.data.from_pandas(pd.read_csv(preprocess_config.test_path), source=preprocess_config.test_path)
            validation_dataset: PandasDataset = mlflow.data.from_pandas(pd.read_csv(preprocess_config.app_validation_path), source=preprocess_config.app_validation_path)

            # Log the datasets to MLflow
            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(test_dataset, context="testing")
            mlflow.log_input(validation_dataset, context="validation")
            log.info(" ---------- Step 2: Data Transformation completed successfully ----------")

            # Model Training
            log.info(" ---------- Step 3: Starting Model Training ----------")
            trainer_config = ModelTrainerConfig()
            trainer = ModelTrainer(config=trainer_config)
            train_df = pd.read_csv(preprocess_config.train_path)
            trained_model = trainer.model_training(train_data=train_df)
            log.info(" ---------- Step 3: Model Training completed successfully ----------")

            # Model Evaluation
            log.info(" ---------- Step 4: Starting Model Evaluation ----------")
            test_df = pd.read_csv(preprocess_config.test_path)
            results = evaluate_model(trained_model, test_df)

            # log the evaluation metrics to MLflow
            mlflow.log_metric("accuracy", results['accuracy'])
            mlflow.log_metric("recall", results['recall'])
            mlflow.log_metric("precision", results['precision'])
            mlflow.log_metric("f1", results['f1'])
            mlflow.log_metric("auc_roc", results['auc_roc'])

            # Log confusion matrix as an artifact
            mlflow.log_artifact("artifacts/confusion_matrix.png", artifact_path="plots")
            mlflow.log_artifact("artifacts/shap_summary.png", artifact_path="plots")
            log.info(" ---------- Step 4: Model Evaluation completed successfully ----------")

            # Log the trained model to MLflow
            mlflow.sklearn.log_model(trained_model, "model")
            log.info(f"  ------ Pipeline completed successfully for Experiment: {experiment_name} Run ID: {mlflow.active_run().info.run_id} ------")

        except Exception as e:
            log.error(f"Pipeline failed: {e}")
            raise CustomException(e, sys)



if __name__ == "__main__":
    # Set up argument parser for command-line execution
    p = argparse.ArgumentParser(description="Credit Customer Churn Prediction Pipeline with MLflow Tracking")

    p.add_argument("--data", type=str, default="data/BankChurners.csv", help="Path to original data")

    p.add_argument("--exp", type=str, default="Churn_Prediction_Experiment", help="MLflow experiment name")

    args = p.parse_args()
    main(data_path=args.data, experiment_name=args.exp)


   
    # Example run command::
    # python main.py --data data/BankChurners.csv --exp Churn_Prediction_Experiment
    
    # To see the MLflow UI, run the following command in terminal and navigate to http://localhost:5000
    # mlflow ui
