import sys
from src.utils.file_logs import get_logger

logger = get_logger()

class CustomException(Exception):
    def __init__(self, message, error_detail):
        super().__init__(message)
        # Extract traceback object
        _, _, exc_tb = error_detail.exc_info()
        # Get the filename and line number
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.line_number = exc_tb.tb_lineno
        self.error_message = message

    def __str__(self):
        # Return a fully formatted message so raising/printing the exception is always informative.
        error_text = (
            f"Error in [{self.file_name}] "
            f"at line [{self.line_number}]: "
            f"{self.error_message}"
        )
        # Log the error details
        logger.error(error_text)
        return error_text