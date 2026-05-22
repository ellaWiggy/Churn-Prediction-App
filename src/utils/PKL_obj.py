# Pickle helpers for saving and loading Python objects
import os
import pickle
import sys

# Custom Exception import
from src.utils.exceptions import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(f"Error saving object to {file_path}: {e}", sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(f"Error loading object from {file_path}: {e}", sys)