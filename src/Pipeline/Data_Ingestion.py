# Usual imports
import pandas as pd
import sys
import os

# Custom imports
from src.utils.file_logs import get_logger
from src.utils.exceptions import CustomException

class DataIngestionConfig:
    data_file_path: str = os.path.join('artifacts', 'original.csv')

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.log = get_logger()
        self.config = config

    def load_data(self, file_path):
        # Check if the file exists
        if not os.path.exists(file_path):
            raise CustomException(f"File not found: {file_path}", sys)
        
        try:
            # If the file exists, load it into a DataFrame
            self.log.info(f"Loading data from {file_path}...")
            data = pd.read_csv(file_path)

            self.log.info(f"Data loaded successfully from {file_path}")
            self.log.info(f"Data into DataFrame with shape: {data.shape}")
            df = pd.DataFrame(data)

            # set CLIENTNUM as index for searching for specific customers in the future
            self.log.info("Setting CLIENTNUM as index...")
            df.set_index('CLIENTNUM', inplace=True)

            # Drop two extra columns that are unknown and useless
            self.log.info("Dropping two columns that start with'Naive_Bayes_Classifier'...")
            df.drop(columns = ['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis = 1, inplace = True) 
            self.log.info("Columns dropped successfully.")
            
            # Save the original data to artifacts/original.csv for reference
            os.makedirs('artifacts', exist_ok=True)
            df.to_csv(self.config.data_file_path, index=True)

            self.log.info(f"Original data saved to {self.config.data_file_path}")

            # Return the loaded DataFrame
            return df
        
        except Exception as e:
            raise CustomException(f"Error loading data from {file_path}: {str(e)}", sys)
       