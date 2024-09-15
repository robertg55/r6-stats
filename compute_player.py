import pandas as pd
import uuid
import argparse
import os
import time
import multiprocessing
import shutil
from typing import Generator, Optional
from log_sorter import sort_file
from log_util import (
    get_files_in_current_directory,
    get_previous_day,
    signal_work_queue_finished,
    check_folder_exists_in_current_directory,
    check_file_exists_in_given_directory,
    setup_logger,
    get_folders_in_current_directory,
    delete_directory,
)

logger = setup_logger()


class LogPlayerBatchParser:
    def __init__(
        self,
        file_path: str,
        batch_size: int = 100000,
    ):
        """
        Initializes the LogPlayerBatchParser.

        :param file_path: The file path to the log file to be parsed.
        :param batch_size: The number of log entries to process per batch. Default is 100,000.
        """
        self.file_path = file_path
        self.batch_size = batch_size
        self.max_kills = 45  # 9 max rounds for ranked with 1 players in each team killing all 5 enemies but going overtime for losing objective

    def parse_logs_from_file_in_player_batches(
        self, file_path: str
    ) -> Generator[list[str], None, None]:
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
                if line_number % 100000 == 0:
                    logger.info(
                        f"processing {file_path} line number {line_number} with elapsed seconds {time.time()-start_time}"
                    )
                if self.is_valid_row(line):
                    # Split each line by commas to create batch
                    new_line = line.strip().split(",")
                    if last_player_id is None:
                        # Handles first row
                        last_player_id = new_line[0]
                    if (
                        split_batch_trigger
                        and last_player_id != new_line[0]
                        and batch != []
                    ):
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


def load_player_top_10_from_path(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads the top 10 player statistics from a file.

    This function reads a file where each line represents a player's match statistics. It parses the data into a
    DataFrame where each row contains a player's match ID and the number of kills in that match.

    :param file_path: The path to the file containing player statistics.
    :return: A DataFrame containing the player's match statistics or None if the file is empty.
    """
    data = []
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


def combine_files_for_previous_days(folders: list[str]) -> None:
    """
    Combines player top 10 files from previous days into a single folder.

    This function creates a directory for the combined files and copies player top 10 files from the specified folders.
    It ensures that files are named properly with the current date and deletes the temporary folders after combining.

    :param folders: A list of folder names containing player top 10 files.
    """
    current_date = folders[-1][-8:]
    all_files_folder_name = f"all_player_top10_{current_date}"
    os.makedirs(all_files_folder_name, exist_ok=True)
    for folder in folders:
        for file in os.listdir(folder):
            source_file = os.path.join(folder, file)
            destination_file = os.path.join(
                all_files_folder_name, f"{file[:-4]}_{current_date}.txt"
            )
            shutil.copy2(source_file, destination_file)

    combined_folder_name = f"combined_player_top10_{current_date}"
    os.makedirs(combined_folder_name, exist_ok=True)
    previous_player = None
    batch = []
    for file in os.listdir(all_files_folder_name):
        player_id = file.split("_")[0]
        if previous_player is None:
            # Handles first file
            previous_player = player_id
            batch = [file]
        else:
            if player_id == previous_player:
                batch += [file]
            else:
                combined_df = combine_batch(batch, all_files_folder_name)
                write_player_top_10_to_file(
                    combined_df, os.path.join(combined_folder_name, file)
                )
                batch = [file]
    # Handles last file
    combined_df = combine_batch(batch, all_files_folder_name)
    write_player_top_10_to_file(
        combined_df, os.path.join(combined_folder_name, batch[0])
    )
    combine_write_player_top_10_to_file(
        combined_folder_name, f"player_top10_{current_date}.txt"
    )
    delete_directory(combined_folder_name)
    delete_directory(all_files_folder_name)


def combine_batch(file_batch: list[str], file_directory: str) -> pd.DataFrame:
    """
    Combines statistics from multiple files into a single DataFrame.

    This function reads a batch of files from a directory, loads the player statistics from each file, and combines them
    into a single DataFrame.

    :param file_batch: A list of filenames containing player statistics.
    :param file_directory: The directory where the files are located.
    :return: A DataFrame containing the combined player statistics.
    """
    combined_stats = None
    for file in file_batch:
        player_stats = load_player_top_10_from_path(os.path.join(file_directory, file))
        if combined_stats is None:
            combined_stats = player_stats
        else:
            combined_stats = concat_statistics_player_top_10(
                combined_stats, player_stats
            )
    return combined_stats


def combine_write_player_top_10_to_file(input_directory: str, output_file: str) -> None:
    """
    Combines player top 10 statistics from multiple files and writes them to a single output file.

    This function reads all text files in the input directory, concatenates their contents, and writes the result
    to the specified output file.

    :param input_directory: The directory containing the input text files.
    :param output_file: The path to the output file where the combined statistics will be written.
    """
    with open(output_file, "w") as outfile:
        # Iterate over all files in the input directory
        for filename in os.listdir(input_directory):
            # Check if the file has a .txt extension
            if filename.endswith(".txt"):
                file_path = os.path.join(input_directory, filename)
                # Open and read the content of each text file
                with open(file_path, "r") as infile:
                    outfile.write(infile.read())


def write_player_top_10(df: pd.DataFrame, date: str, player_id: str) -> None:
    """
    Writes the top 10 player match statistics to a file for the given player.

    This function writes the player's top 10 match statistics to a file in the format `player_top10_<date>/<player_id>.txt`.

    :param df: A DataFrame containing the player's top 10 match statistics.
    :param date: The date of the matches.
    :param player_id: The player ID for whom the statistics are written.
    """
    folder = f"player_top10_{date}"
    file_path = os.path.join(folder, f"{player_id}.txt")
    write_player_top_10_to_file(df, file_path)


def write_player_top_10_to_file(df: pd.DataFrame, file_path: str) -> None:
    """
    Writes the player's top 10 match statistics to a file.

    This function takes a DataFrame containing player statistics and writes the top 10 matches by kills to a specified
    file in a specific format.

    :param df: A DataFrame containing the player's top 10 match statistics.
    :param file_path: The path to the file where the statistics will be written.
    """
    with open(file_path, "w") as file:
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


def concat_statistics_player_top_10(
    original_df: pd.DataFrame, new_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
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
    player_id = df["player_id"].iloc[0]
    player_top_10 = compute_top_10_player_matches(df)
    write_player_top_10(player_top_10, date, player_id)


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


def generate_statistics(
    files_to_consume: Optional[str],
    batch_size: int,
    worker_count: int,
    number_of_past_days: int,
) -> None:
    """
    Generates player top 10 match statistics for the specified files.

    This function processes log files for the past `number_of_past_days`, sorts the files by player, and computes the
    top 10 matches by kills for each player using multiprocessing. It also combines statistics from previous days into
    a single output.

    :param files_to_consume: A comma-separated string of filenames to process, or None to process all available files.
    :param batch_size: The number of log entries to process per batch.
    :param worker_count: The number of worker processes to use for parallel processing.
    :param number_of_past_days: The number of past days to include in the analysis.
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
        if check_folder_exists_in_current_directory(f"player_top10_{date}"):
            # if it was already previously computed or the statistic generation is disabled
            logger.info(f"already processed {file_path}, skipping")
            continue
        os.makedirs(f"player_top10_{date}", exist_ok=True)
        sorted_file_path = f"player-sorted-{file_path}"
        sort_file(file_path, sorted_file_path, "player")
        # Create a multiprocessing Queue to store work tasks in a thread-safe way
        work_queue = multiprocessing.Queue()
        workers = []
        for _ in range(worker_count):
            p = multiprocessing.Process(target=worker, args=(work_queue, date))
            workers.append(p)
            p.start()
        parser = LogPlayerBatchParser(
            sorted_file_path,
            batch_size=batch_size,
        )
        log_parser_generator = parser.parse_logs_from_file_in_player_batches(file_path)
        next_batch = None
        while True:
            if work_queue.qsize() < worker_count * WORKER_CACHE_MULTIPLIER:
                next_batch = next(log_parser_generator, None)
                work_queue.put(next_batch)
                if next_batch is None:
                    signal_work_queue_finished(work_queue)
                    break
        for worker_process in workers:
            # wait for all workers to finish
            worker_process.join()

    folders_in_directory = get_folders_in_current_directory()
    daily_folders = [
        folder for folder in folders_in_directory if folder.startswith("player_top10_")
    ]
    sorted_daily_folders = sorted(daily_folders)[-number_of_past_days:]
    combine_files_for_previous_days(sorted_daily_folders)
    logger.info("finished computing player top 10 matches by kill")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computer player top 10 matches by kill for R6 matches."
    )
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

    generate_statistics(
        args.files_to_consume, args.batch_size, args.workers, args.number_of_past_days
    )
