# Usual imports
import pandas as pd
import sys
import os
import json

# Custom imports
from src.utils.file_logs import get_logger
from src.utils.exceptions import CustomException
from src.utils.PKL_obj import save_object
# Train test split
from sklearn.model_selection import train_test_split

# Machine Learning imports.
from lightgbm import LGBMClassifier

# Preprocessing.
from sklearn.pipeline import Pipeline
from src.utils.Modeling import (
    FeatureEngineering,
    feature_selector
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Silence warnings for cleaner logs
import logging
import warnings
warnings.filterwarnings('ignore')
logging.getLogger('lightgbm').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.ERROR)


# Set path for train and test data to be saved in artifacts folder
class PreprocessConfig:
    train_path: str = os.path.join('artifacts', 'train.csv')
    test_path: str = os.path.join('artifacts', 'test.csv')
    app_validation_path: str = os.path.join('artifacts', 'app_validation.csv')
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    model_results_path: str = os.path.join('data', 'modeling_results.json')
    

# Process class for data preprocessing
class Preprocess:
    def __init__(self, config: PreprocessConfig):
        self.config = config
        # Initialize logger
        self.log = get_logger()

    def process_data(self, df: pd.DataFrame):
       
        try:
            
            self.log.info("Starting data processing...")
            # Make all columns lowercase
            df.columns= [x.lower() for x in df.columns]

            # Change name of target column to churn
            df.rename(columns={'attrition_flag': 'churn'}, inplace = True)

            # Map target column to binary values
            df['churn'] = df['churn'].map({'Attrited Customer': 1, 'Existing Customer': 0})

            self.log.info("Drop any columns that are not needed and those that are not under ECOA standards...")
            # Drop columns that are not needed for modeling
            df.drop(columns=['customer_age', 'gender', 'avg_open_to_buy'], axis=1, inplace=True)
            self.log.info("Columns, [customer_age, gender, avg_open_to_buy], dropped successfully.")
            
            # Creating train, test, and validation splits from dataframe
            self.log.info("Creating train, test, and validation splits from dataframe...")
            df_dev, df_validation = train_test_split(df, test_size=0.10, random_state=42, stratify=df['churn'])
            
            # Split the df_dev data into X and y for train and test sets and drop y from df_validation set
            X = df_dev.drop('churn', axis=1)
            y = df_dev['churn'].copy()
            df_validation = df_validation.drop('churn', axis=1)

            # Split the data into training and testing sets using X and y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Log the shapes of the train, test, and validation sets for better traceability
            self.log.info("X_train Shape: {}, y_train Shape: {}".format(X_train.shape, y_train.shape))
            self.log.info("X_test Shape: {}, y_test Shape: {}".format(X_test.shape, y_test.shape))
            self.log.info("Validation Shape: {}".format(df_validation.shape))

            # Make sure the artifacts directory exists before trying to save files
            os.makedirs('artifacts', exist_ok=True)

            # Concatenate X and y for train and test sets before saving to CSV
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)

            # Save the train, test, and validation data to CSV files in the artifacts folder
            train_data.to_csv(self.config.train_path, index=False)
            test_data.to_csv(self.config.test_path, index=False)
            df_validation.to_csv(self.config.app_validation_path, index=True)

            # Log the successful processing of the data
            self.log.info(f"Data processed successfully.")
            self.log.info(f"Train data saved to {self.config.train_path}, test data saved to {self.config.test_path}, and validation data saved to {self.config.app_validation_path}")

        except Exception as e:
            raise CustomException(f"Error processing data: {str(e)}", sys)

    def get_preprocessor_pipeline(self):
        # Creates the preprocessor for data transformation
        try:
            self.log.info("Starting data transformation object creation.")
            self.log.info("Load in Best Parameters found in EDA_Modeling notebook for LGBMClassifier.")
            clean_results = {}
            best_min_features = 21
            # Check to see if file exists before trying to load it
            if os.path.exists(self.config.model_results_path):
                with open(self.config.model_results_path, 'r') as f:
                    config = json.load(f)
                    
                clean_results = config.get('params', {})
                best_min_features = config.get('min_features', 21)
                self.log.info("Successfully loaded saved parameters!")
            else:
                self.log.warning(f"JSON not found at {self.config.model_results_path}. Using default values.")

            #feature setup
            one_hot_features = ['marital_status']
            ordinal_features = {
                'education_level': ['Uneducated',
                                    'High School',
                                    'College',
                                    'Graduate',
                                    'Post-Graduate',
                                    'Doctorate',
                                    'Unknown'],
                'income_category': ['Less than $40K',
                                    '$40K - $60K',
                                    '$60K - $80K',
                                    '$80K - $120K',
                                    '$120K +',
                                    'Unknown'],
                'card_category': ['Blue',
                                'Silver',
                                'Gold',
                                'Platinum']
            }
            # Log the feature engineering and transformation steps for better traceability
            self.log.info(f'Categorical features: {list(ordinal_features.keys())}.')
            self.log.info(f'Ordinal encoding applied to: {list(ordinal_features.keys())}.')
            self.log.info(f'One-hot encoding applied to: {one_hot_features}.')

            # Create the ColumnTransformer with the transformations for ordinal and one-hot features, 
            # and passthrough for the rest 
            transformer = ColumnTransformer(
                transformers=[
                    ('ordinal', OrdinalEncoder(categories=list(ordinal_features.values())), list(ordinal_features.keys())),
                    ('one_hot', OneHotEncoder(drop=None, sparse_output=False), one_hot_features),
                    ],remainder='passthrough').set_output(transform="pandas")
                        
            # Create the full pipeline with feature engineering, 
            # transformation, 
            # feature selection, 
            # and model using the best parameters found in the notebook
            pipeline = Pipeline([
                ('engineering', FeatureEngineering()),
                ('transformation', transformer),
                ('selection', feature_selector(
                    estimator=LGBMClassifier(random_state=42, verbosity=-1), 
                    min_features_to_select=best_min_features
                )),
                ('model', LGBMClassifier(
                    random_state=42, 
                    **clean_results))
            ]).set_output(transform="pandas")

            # Save the preprocessor object to a file using the save_object function from PKL_obj.py
            save_object(file_path= self.config.preprocessor_obj_file_path, obj=pipeline)
            self.log.info(f"Preprocessor Pipeline saved to {self.config.preprocessor_obj_file_path}")
            
            # Return the pipeline object for use in the model trainer
            return pipeline
        
        except Exception as e:
            self.log.error(f"Error in creating data transformer object: {e}")
            raise CustomException(e, sys)
