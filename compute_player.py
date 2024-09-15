import pandas as pd
import uuid
import argparse
import os
import time
import multiprocessing
import shutil
from typing import Generator, Optional
from log_sorter import sort_file
from log_util import get_files_in_current_directory, get_previous_day, signal_work_queue_finished, check_folder_exists_in_current_directory, check_file_exists_in_given_directory, setup_logger

logger = setup_logger()


class LogPlayerBatchParser:
    def __init__(self, file_path: str, batch_size: int = 100000,):
        """
        Initializes the LogPlayerBatchParser.

        :param file_path: The file path to the log file to be parsed.
        :param batch_size: The number of log entries to process per batch. Default is 100,000.
        """
        self.file_path = file_path
        self.batch_size = batch_size
        self.max_kills = 45 # 9 max rounds for ranked with 1 players in each team killing all 5 enemies but going overtime for losing objective

    def parse_logs_from_file_in_player_batches(self, file_path: str) -> Generator[list[str], None, None]:
        """
        Parses the log file and yields batches of log entries grouped by player ID.

        This method reads the log file and splits the log entries into batches, ensuring that the batch doesn't
        split in the middle of a player's match statistics.

        :param file_path: The path to the log file to parse.
        :yield: A list representing a batch of log entries for a specific player.
        """
        # Read the file contents
        last_player_id = None
        split_batch_trigger = False
        start_time = time.time() 
        with open(file_path, "r") as file:
            batch = []
            for line_number, line in enumerate(file, start=1):
                if line_number%100000==0:
                    logger.info(f"processing {file_path} line number {line_number} with elapsed seconds {time.time()-start_time}")
                if self.is_valid_row(line):
                    # Split each line by commas to create batch
                    new_line = line.strip().split(",")
                    if last_player_id is None:
                        # Handles first row
                        last_player_id = new_line[1]
                    if split_batch_trigger and last_player_id != new_line[0] and batch != []:
                        # Doesn't split the batch in the middle of a match statistic
                        split_batch_trigger = False
                        last_player_id = new_line[0]
                        yield batch  # Yield the batch when batch_size is reached
                        batch = []  # Reset the batch
                    # Add line to batch
                    batch.append(new_line)
                if line_number % self.batch_size == 0:
                    # Trigger split batch on invalid rows too
                    split_batch_trigger = True
                
            if batch:
                yield batch  # Yield the remaining lines if there are any

    def is_valid_row(self, row: str) -> bool:
        """
        Validates whether a row from the log file has the correct format and valid data.

        This method checks if the row contains valid UUIDs for `player_id` and `match_id`, valid integers for
        `operator_id` and `nb_kills`, and that the number of kills does not exceed the maximum allowed.

        :param row: A string representing a log entry row.
        :return: True if the row is valid, False otherwise.
        """
        try:
            player_id, match_id, operator_id, nb_kills = row.split(",")
            uuid.UUID(player_id)  # Validate UUID
            uuid.UUID(match_id)  # Validate UUID
            int(operator_id)  # Validate operator_id as integer
            kills = int(nb_kills)  # Validate nb_kills as integer
            if kills > self.max_kills:
                return False
            return True
        except Exception:
            return False


def compute_top_10_player_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the top 10 player matches by the total number of kills.

    This function groups the DataFrame by `player_id` and `match_id`, sums the number of kills for each player in each match,
    and retrieves the top 10 matches based on the number of kills.

    :param df: A DataFrame containing player match data.
    :return: A DataFrame containing the top 10 player matches by total kills.
    """
    grouped_df = df.groupby(["player_id", "match_id"])["nb_kills"].sum().reset_index()
    return get_top_10_avg_kill_per_player_id(grouped_df)


def get_top_10_avg_kill_per_player_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieves the top 10 player matches by the number of kills.

    This function groups the input DataFrame by `player_id`, sorts the matches by kills in descending order, and returns
    the top 10 matches for each player.

    :param df: A DataFrame containing player match data.
    :return: A DataFrame containing the top 10 matches for each player.
    """
    return (
        df.groupby("player_id")
        .apply(lambda group: group.sort_values(by="nb_kills", ascending=False).head(10))
        .reset_index(drop=True)
    )

def load_player_top_10_from_previous_day(current_date: str, player_id: str) -> Optional[pd.DataFrame]:
    """
    Loads the top 10 player statistics from the previous day's file, if it exists.

    This function attempts to load a player's top 10 statistics from the previous day's file. If the file is found,
    the data is returned as a DataFrame; otherwise, `None` is returned.

    :param current_date: The current date in 'YYYYMMDD' format.
    :param player_id: The player ID for which to load the statistics.
    :return: A DataFrame containing the player's top 10 match statistics from the previous day, or None if not found.
    """

    folder_name = f"player_top10_{get_previous_day(current_date)}"
    file_name = f"{player_id}.txt"
    if check_folder_exists_in_current_directory(folder_name) and check_file_exists_in_given_directory(folder_name, file_name):
            data = []
            file_path = os.path.join(os.getcwd(), folder_name, file_name)
            with open(file_path, "r") as file:
                for line in file:
                    player_id, matches_kills = line.strip().split("|")
                    # Parse the matches_kills inline
                    for mk in matches_kills.split(","):
                        match_id, nb_kills = mk.split(":")
                        data.append(
                            {
                                "player_id": player_id,
                                "match_id": match_id,
                                "nb_kills": int(nb_kills),
                            }
                        )
            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(data)
            return df
    return None

def add_missing_statistics_from_previous_day(date: str) -> None:
    """
    Adds missing player statistics from the previous day's data.

    This function checks if any player statistics from the previous day are missing in the current day's data,
    and copies the missing files into the current day's folder.

    :param date: The current date in 'YYYYMMDD' format.
    """
    previous_folder_name = f"player_top10_{get_previous_day(date)}"
    current_folder_name = f"player_top10_{date}"
    if check_folder_exists_in_current_directory(previous_folder_name):
        previous_path = os.path.join(os.getcwd(), previous_folder_name)
        
        previous_files = [
            file
            for file in os.listdir(previous_path)
            if os.path.isfile(os.path.join(previous_path, file))
        ]
        current_path = os.path.join(os.getcwd(), current_folder_name)
        current_files = [
            file
            for file in os.listdir(current_path)
            if os.path.isfile(os.path.join(current_path, file))
        ]
        missing_files = list(set(previous_files) - set(current_files))
        for missing_file in missing_files:
            source_file = os.path.join(previous_path, missing_file)
            destination_file = os.path.join(current_path, missing_file)
            shutil.copy2(source_file, destination_file)
            

def combine_write_player_top_10_to_file(date: str) -> None:
    """
    Combines individual player top 10 files into a single file for the given date.

    This function reads all player top 10 files for the given date and combines them into a single file
    named `player_top10_<date>.txt`.

    :param date: The date for which to combine player statistics.
    """
    input_directory = f"player_top10_{date}"
    with open(f"player_top10_{date}.txt", 'w') as outfile:
        # Iterate over all files in the input directory
        for filename in os.listdir(input_directory):
            # Check if the file has a .txt extension
            if filename.endswith('.txt'):
                file_path = os.path.join(input_directory, filename)
                # Open and read the content of each text file
                with open(file_path, 'r') as infile:
                    outfile.write(infile.read())

def write_player_top_10_to_file(df: pd.DataFrame, date: str, player_id: str) -> None:
    """
    Writes the top 10 player match statistics to a file for the given player.

    This function writes the player's top 10 match statistics to a file in the format `player_top10_<date>/<player_id>.txt`.

    :param df: A DataFrame containing the player's top 10 match statistics.
    :param date: The date of the matches.
    :param player_id: The player ID for whom the statistics are written.
    """
    folder = f"player_top10_{date}"
    file_path = os.path.join(folder, f'{player_id}.txt')
    with open(file_path, 'w') as file:
        result = (
            df.groupby("player_id")
            .apply(
                lambda x: ",".join(
                    [f"{row['match_id']}:{row['nb_kills']}" for _, row in x.iterrows()]
                )
            )
            .reset_index()
        )

        # Rename the column to make it more clear
        result.columns = ["player_id", "matches_kills"]

        # Display the result
        for _, row in result.iterrows():
            file.write(f"{row['player_id']}|{row['matches_kills']}\n")


def concat_statistics_player_top_10(original_df: pd.DataFrame, new_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates two DataFrames containing player statistics and returns the top 10 matches by kills.

    This function combines the original DataFrame with a new DataFrame of player statistics, and returns the top 10
    player matches by kills.

    :param original_df: The original DataFrame containing player statistics.
    :param new_df: The new DataFrame to concatenate with the original one. If None, the original DataFrame is returned.
    :return: A DataFrame containing the top 10 player matches by kills.
    """
    if new_df is None:
        return original_df
    return get_top_10_avg_kill_per_player_id(pd.concat([original_df, new_df]))


def process_players_batch(batch: list[list[str]], date: str) -> None:
    """
    Processes a batch of player match data and computes the top 10 matches by kills.

    This function takes a batch of player match data, processes each player's matches, and computes the top 10 matches
    by kills for each player.

    :param batch: A list of log entries representing player match data.
    :param date: The date of the matches.
    """
        # Create a DataFrame from the parsed log data
    df = pd.DataFrame(
        batch, columns=["player_id", "match_id", "operator_id", "nb_kills"]
    )

    # Convert 'operator_id' and 'nb_kills' columns to integers
    df["operator_id"] = df["operator_id"].astype(int)
    df["nb_kills"] = df["nb_kills"].astype(int)

    # Get unique player IDs
    unique_player_ids = df["player_id"].unique()

    # Process each player individually
    for player_id in unique_player_ids:
        # Filter the dataframe for the current player
        player_df = df[df["player_id"] == player_id]
        process_player_batch(player_df, date)

def process_player_batch(df: pd.DataFrame, date: str) -> None:
    """
    Processes a single player's match data and computes the top 10 matches by kills.

    This function takes a player's match data, computes the top 10 matches by kills, and updates the player's
    top 10 statistics.

    :param df: A DataFrame containing the player's match data.
    :param date: The date of the matches.
    """
    player_id = df['player_id'].iloc[0]
    player_top_10 = compute_top_10_player_matches(df)
    previous_day_player_top_10 = load_player_top_10_from_previous_day(date, player_id)
    combined_player_top_10 = concat_statistics_player_top_10(player_top_10, previous_day_player_top_10)
    write_player_top_10_to_file(combined_player_top_10, date, player_id)

def worker(work_queue: multiprocessing.Queue, date: str) -> None:
    """
    Worker function to process batches of player match data in parallel.

    This function continuously retrieves tasks from the work queue, processes them, and signals when all tasks are completed.

    :param work_queue: A multiprocessing queue containing batches of player match data.
    :param date: The date of the matches.
    """
    while True:
        task = work_queue.get()
        if task is None:
            break
        process_players_batch(task, date)
    signal_work_queue_finished(work_queue)

def generate_statistics(files_to_consume: Optional[str], batch_size: int, worker_count: int, number_of_past_days: int) -> None:
    """
    Generates player statistics from R6 Siege match logs, computing the top 10 player matches by total kills.

    This function processes log files to compute the top 10 player matches based on the total number of kills. It sorts the log files, 
    processes them in batches, and uses multiple worker processes for parallel computation. If previously computed statistics exist for 
    a given date, those files are skipped. It also handles adding missing statistics from previous days and combines all player statistics 
    into a single output file.

    :param files_to_consume: A comma-separated string of file names to process. If not provided, the function processes 
                             all files in the current directory that start with 'r6-matches-'.
    :param batch_size: The number of log entries to process per batch for memory optimization.
    :param worker_count: The number of worker processes to use for parallel processing.
    :param number_of_past_days: The number of past days' logs to process, using the most recent files.

    :return: None

    This function:
    - Sorts log files by player before processing.
    - Uses multiprocessing to distribute workload across multiple workers.
    - Processes batches of log entries in a memory-efficient manner.
    - Skips files that have already been processed.
    - Adds missing player statistics from the previous day's data.
    - Combines the processed player top 10 matches into a single file for each day.
    """

    WORKER_CACHE_MULTIPLIER = 8
    files_in_directory = get_files_in_current_directory()
    if not files_to_consume:
        r6_files = [
            file for file in files_in_directory if file.startswith("r6-matches-")
        ]
    else:
        r6_files = files_to_consume.split(",")
    sorted_r6_files = sorted(r6_files)[-number_of_past_days:]
    logger.info(f"processing player top 10 matches by kill for files {sorted_r6_files}")
    for file_path in sorted_r6_files:
        date = file_path[11:-4]
        if f"player_top10_{date}.txt" in files_in_directory:
            # if it was already previously computed or the statistic generation is disabled
            logger.info(f"already processed {file_path}, skipping")
            continue
        os.makedirs(f"player_top10_{date}", exist_ok=True)
        sorted_file_path = "player-sorted-"+file_path
        sort_file(file_path, sorted_file_path, "player")
        # Create a multiprocessing Queue to store work tasks in a thread-safe way
        work_queue = multiprocessing.Queue()
        workers = []
        for _ in range(worker_count):
            p = multiprocessing.Process(target=worker, args=(work_queue,date))
            workers.append(p)
            p.start()
        parser = LogPlayerBatchParser(sorted_file_path, batch_size=batch_size,)
        log_parser_generator = parser.parse_logs_from_file_in_player_batches(file_path)
        next_batch = None
        while True:
            if work_queue.qsize() < worker_count*WORKER_CACHE_MULTIPLIER:
                next_batch = next(log_parser_generator, None)
                work_queue.put(next_batch)
                if next_batch is None:
                    signal_work_queue_finished(work_queue)
                    break
        for worker_process in workers:
            # wait for all workers to finish
            worker_process.join()
        
        add_missing_statistics_from_previous_day(date)
        combine_write_player_top_10_to_file(date)
    logger.info("finished computing player top 10 matches by kill")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computer player top 10 matches by kill for R6 matches.")
    parser.add_argument(
        "--files-to-consume",
        type=str,
        help="Name of files to consume,comma separated, will process everything if not provided",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for memory optimization",
    )
    parser.add_argument(
        "--number-of-past-days",
        type=int,
        default=7,
        help="Only consumes this many days in the past",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of separate workers running in parallel",
    )

    args = parser.parse_args()

    generate_statistics(args.files_to_consume, args.batch_size, args.workers, args.number_of_past_days)
