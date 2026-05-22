# Usual imports
import pandas as pd
import sys
import os
import json

# Custom imports for logging, exceptions, and saving/loading objects
from src.utils.file_logs import get_logger
from src.utils.exceptions import CustomException
from src.utils.PKL_obj import save_object
from src.utils.PKL_obj import load_object

# Preprocessing.
from src.Pipeline.Data_Transformation import Preprocess, PreprocessConfig

# Silence warnings for cleaner logs
import logging
import warnings
warnings.filterwarnings('ignore')
logging.getLogger('lightgbm').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.ERROR)

class ModelTrainerConfig:
    model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        # Initialize logger
        self.log = get_logger()

    def model_training(self, train_data):
        try:
            # Build the preprocessing pipeline using the preprocessing config.
            self.log.info("Building preprocessor pipeline...")
            pipeline = Preprocess(config=PreprocessConfig()).get_preprocessor_pipeline()
            if pipeline is None:
                self.log.error("Preprocessor pipeline could not be loaded.")
            else:
                self.log.info("Preprocessor object loaded successfully.")

            # Separate features and target variable from the training data
            X_train = train_data.drop(columns=['churn'], axis=1)
            y_train = train_data['churn']
            
            # Train the model using the pipeline
            self.log.info("Training the model...")
            trained_model = pipeline.fit(X_train, y_train)
            
            # Save the trained Model
            self.log.info("Saving Final Pipeline with trained model...")
            save_object(file_path = self.config.model_file_path, obj = trained_model)
            self.log.info(f"Trained model pipeline saved to {self.config.model_file_path}")

            return trained_model
        
        except Exception as e:
            raise CustomException(f"Error during model training: {str(e)}", sys)