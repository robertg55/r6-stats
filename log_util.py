import os
import multiprocessing
import logging
import shutil
import sys
from datetime import datetime


def setup_logger() -> logging.Logger:
    """
    Sets up and configures a logger for the application.

    This function creates a logger that logs messages to both `stdout` and a log file named with the current
    date and time in the format 'app_YYYYMMDD_HHMMSS.log' in a 'logs' directory. It sets the log level to `DEBUG`
    and ensures that the logger does not add multiple handlers if it has already been set up.

    If the 'logs' directory does not exist, it is created.

    :return: A configured logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Avoid adding multiple handlers if the logger already has handlers
    if not logger.hasHandlers():
        # Create a handler for stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Generate a log file name with the current date and time inside the logs folder
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"app_{current_time}.log")

        # Create a handler for logging to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s]: %(message)s")
        stdout_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(stdout_handler)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()


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


def delete_directory(directory_path: str) -> None:
    """
    Deletes the specified directory if it exists. Logs an error if the directory does not exist
    or if an error occurs during deletion.

    :param directory_path: The path to the directory to be deleted.
    """
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
        except Exception as e:
            logger.error(f"Error occurred while deleting directory: {e}")
    else:
        logger.error(f"Directory '{directory_path}' does not exist.")


def get_folders_in_current_directory() -> list[str]:
    """
    Retrieves a list of all folders in the current working directory.

    This function scans the current directory and returns only the folders, excluding files and other
    non-directory items.

    :return: A list of folder names found in the current working directory.
    """
    current_directory = os.getcwd()
    return [
        folder
        for folder in os.listdir(current_directory)
        if os.path.isdir(os.path.join(current_directory, folder))
    ]

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


def signal_work_queue_finished(work_queue: multiprocessing.Queue) -> None:
    """
    Signals to the workers that the work queue has been finished by putting `None` in the queue.

    This function sends 100 `None` values to the queue, which serves as a signal for the workers to stop.
    Multiple signals are sent to account for any synchronization issues that may arise during multiprocessing.

    :param work_queue: A `multiprocessing.Queue` object where tasks are placed for workers to process.
    """
    for _ in range(100):
        work_queue.put(
            None
        )  # Signal to the next worker that we're done, sends many signals for sync issues
