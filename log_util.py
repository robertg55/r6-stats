import os
import multiprocessing
import logging
import sys
from datetime import datetime, timedelta

def setup_logger() -> logging.Logger:
    """
    Sets up and configures a logger for the application.

    This function creates a logger that logs messages to both `stdout` and a log file named with the current
    date and time in the format 'app_YYYYMMDD_HHMMSS.log'. It sets the log level to `DEBUG` and ensures that
    the logger does not add multiple handlers if it has already been set up.

    :return: A configured logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a handler for stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)

    # Generate a log file name with the current date and time
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'app_{current_time}.log'

    # Create a handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]: %(message)s')
    stdout_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Avoid adding multiple handlers if the logger is already set up
    if not logger.hasHandlers():
        logger.addHandler(stdout_handler)
        logger.addHandler(file_handler)

    return logger

def get_files_in_current_directory() -> list[str]:
    """
    Retrieves a list of all files in the current working directory.

    This function scans the current directory and returns only the files, excluding directories and other
    non-file items.

    :return: A list of filenames found in the current working directory.
    """
    current_directory = os.getcwd()
    return [
        file
        for file in os.listdir(current_directory)
        if os.path.isfile(os.path.join(current_directory, file))
    ]

def check_file_exists_in_given_directory(folder_name:str, file_name: str) -> bool:
    """
    Checks if a file exists in the current working directory.

    :param file_name: The name of the file to check.
    :return: True if the file exists, False otherwise.
    """
    file_path = os.path.join(os.getcwd(), folder_name, file_name)
    if os.path.isfile(file_path):
        return True
    else:
        return False

def check_folder_exists_in_current_directory(folder_name: str) -> bool:
    """
    Checks if a folder exists in the current working directory.

    :param folder_name: The name of the folder to check.
    :return: True if the folder exists, False otherwise.
    """
    folder_path = os.path.join(os.getcwd(), folder_name)
    if os.path.isdir(folder_path):
        return True
    else:
        return False


def get_previous_day(current_date):
    """
    Calculates the previous day given a current date in 'YYYYMMDD' format.

    This function converts the `current_date` string to a `datetime` object, subtracts one day, and returns
    the previous day's date as a string in the same 'YYYYMMDD' format.

    :param current_date: The current date in 'YYYYMMDD' format.
    :return: The previous day's date in 'YYYYMMDD' format.
    """
    date_obj = datetime.strptime(current_date, "%Y%m%d")
    previous_day = date_obj - timedelta(days=1)
    return previous_day.strftime("%Y%m%d")


def signal_work_queue_finished(work_queue: multiprocessing.Queue) -> None:
    """
    Signals to the workers that the work queue has been finished by putting `None` in the queue.

    This function sends 100 `None` values to the queue, which serves as a signal for the workers to stop. 
    Multiple signals are sent to account for any synchronization issues that may arise during multiprocessing.

    :param work_queue: A `multiprocessing.Queue` object where tasks are placed for workers to process.
    """
    for _ in range(100):
        work_queue.put(None)  # Signal to the next worker that we're done, sends many signals for sync issues
