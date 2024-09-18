import pandas as pd
import uuid
import argparse
import os
import time
import multiprocessing
import shutil
from typing import Generator, Optional
from log_sorter import split_file, CHUNK_SIZE, merge_chunk_files
from log_util import (
    get_files_in_current_directory,
    signal_work_queue_finished,
    check_folder_exists_in_current_directory,
    setup_logger,
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
                if line_number % 1000000 == 0:
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


def merge_text_files(input_directory, output_file):
    """
    Merge all text files from the input_directory into one output_file without loading them all into memory.

    :param input_directory: Directory containing the text files to merge
    :param output_file: Path to the output file
    """
    # Ensure the input directory exists
    if not os.path.isdir(input_directory):
        logger.error(f"Directory {input_directory} does not exist.")
        return

    # Open the output file for writing
    with open(output_file, "w") as outfile:
        # Iterate over all the files in the input directory
        for filename in os.listdir(input_directory):
            file_path = os.path.join(input_directory, filename)

            # Only process text files
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                # Open each input file in read mode and copy its content line by line
                with open(file_path, "r") as infile:
                    for line in infile:
                        outfile.write(line)


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


def process_players_batch(batch: list[list[str]]) -> None:
    """
    Processes a batch of player match data and computes the top 10 matches by kills.

    This function takes a batch of player match data, processes each player's matches, and computes the top 10 matches
    by kills for each player.

    :param batch: A list of log entries representing player match data.
    """
    # Create a DataFrame from the parsed log data
    df = pd.DataFrame(
        batch, columns=["player_id", "match_id", "operator_id", "nb_kills"]
    )

    # Convert 'operator_id' and 'nb_kills' columns to integers
    df["operator_id"] = df["operator_id"].astype(int)
    df["nb_kills"] = df["nb_kills"].astype(int)

    player_top_10 = compute_top_10_player_matches(df)
    write_player_top_10_to_file(
        player_top_10, os.path.join("tmp_join_output_dir", f"{uuid.uuid4()}.txt")
    )


def worker(work_queue: multiprocessing.Queue) -> None:
    """
    Worker function to process batches of player match data in parallel.

    This function continuously retrieves tasks from the work queue, processes them, and signals when all tasks are completed.

    :param work_queue: A multiprocessing queue containing batches of player match data.
    """
    while True:
        task = work_queue.get()
        if task is None:
            break
        process_players_batch(task)
    signal_work_queue_finished(work_queue)


def generate_chunked_sorted_files(
    sorted_r6_files: list[str], tmp_all_files_folder_name: str
) -> tuple[list[str], str]:
    """
    Splits sorted R6 Siege match log files into chunks and stores them in a temporary directory.

    This function processes sorted R6 Siege match files by splitting them into smaller chunks and copying the
    resulting chunk files to a temporary folder for further processing. It returns the names of the chunked files
    and the latest date processed.

    :param sorted_r6_files: List of sorted R6 Siege match log file paths to process.
    :param tmp_all_files_folder_name: Temporary folder name where chunked files will be copied.
    :return: A tuple containing the list of chunked file names and the latest date processed.
    """
    chunked_file_names = []
    latest_date = ""
    for file_path in sorted_r6_files:
        date = file_path[11:-4]
        latest_date = date
        output_dir = f"chunks-player-{date}"
        if not check_folder_exists_in_current_directory(output_dir):
            os.makedirs(output_dir)
            logger.info(f"separating chunk files in {output_dir}")
            split_file(file_path, CHUNK_SIZE, output_dir, True)
        logger.info(f"copying files in f{output_dir} to {tmp_all_files_folder_name}")
        for file in os.listdir(output_dir):
            source_file = os.path.join(output_dir, file)
            destination_file = os.path.join(
                tmp_all_files_folder_name, f"{file[:-4]}_{date}.txt"
            )
            chunked_file_names.append(destination_file)
            shutil.copy2(source_file, destination_file)
    return (chunked_file_names, latest_date)


def process_log_batches_with_workers(
    tmp_join_output_dir: str,
    worker_count: int,
    output_multi_days_file: str,
    batch_size: int,
    latest_date: str,
    worker_cache_multiplier: int,
) -> None:
    """
    Processes log batches using multiple workers and merges the results.

    This function processes player log batches using a specified number of workers. It reads batches of log entries
    from the given output file and assigns them to workers using a multiprocessing queue. After all workers finish
    processing, it merges the output into a single file.

    :param tmp_join_output_dir: Temporary directory to store intermediate outputs before merging.
    :param worker_count: The number of worker processes to use for parallel processing.
    :param output_multi_days_file: The file containing multiple days of logs to process.
    :param batch_size: The number of log entries to process per batch.
    :param latest_date: The latest date for naming the output file.
    :param worker_cache_multiplier: Multiplier for worker queue caching, determining how many batches to queue for each worker.
    """
    os.makedirs(tmp_join_output_dir, exist_ok=True)
    # Create a multiprocessing Queue to store work tasks in a thread-safe way
    work_queue = multiprocessing.Queue()
    workers = []
    for _ in range(worker_count):
        p = multiprocessing.Process(target=worker, args=(work_queue,))
        workers.append(p)
        p.start()
    parser = LogPlayerBatchParser(
        output_multi_days_file,
        batch_size=batch_size,
    )
    log_parser_generator = parser.parse_logs_from_file_in_player_batches(
        output_multi_days_file
    )
    next_batch = None
    while True:
        if work_queue.qsize() < worker_count * worker_cache_multiplier:
            next_batch = next(log_parser_generator, None)
            work_queue.put(next_batch)
            if next_batch is None:
                signal_work_queue_finished(work_queue)
                break
    for worker_process in workers:
        # wait for all workers to finish
        worker_process.join()
    merge_text_files(tmp_join_output_dir, f"player_top10_{latest_date}.txt")


def generate_statistics(
    files_to_consume: Optional[str],
    batch_size: int,
    worker_count: int,
    number_of_past_days: int,
    worker_cache_multiplier: int,
) -> None:
    """
    Generates player top 10 statistics by kills for R6 Siege matches over the specified number of past days.

    This function processes log files for the specified number of past days, splitting them into chunks, merging
    the chunked files, and processing the log batches with multiple workers. After processing, it cleans up the
    temporary files and directories.

    :param files_to_consume: A comma-separated string of filenames to process, or None to process all available files
                             from the current directory matching the pattern "r6-matches-".
    :param batch_size: The number of log entries to process per batch.
    :param worker_count: The number of worker processes to use for parallel processing.
    :param number_of_past_days: The number of past days' worth of log files to include in the analysis.
    :param worker_cache_multiplier: Multiplier for worker queue caching, determining how many batches to queue for each worker.
    """
    files_in_directory = get_files_in_current_directory()
    if not files_to_consume:
        r6_files = [
            file for file in files_in_directory if file.startswith("r6-matches-")
        ]
    else:
        r6_files = files_to_consume.split(",")
    sorted_r6_files = sorted(r6_files)[-number_of_past_days:]
    logger.info(f"processing player top 10 matches by kill for files {sorted_r6_files}")

    tmp_all_files_folder_name = "player_top10_tmp"
    os.makedirs(tmp_all_files_folder_name, exist_ok=True)

    chunked_file_names, latest_date = generate_chunked_sorted_files(
        sorted_r6_files, tmp_all_files_folder_name
    )
    logger.info(f"merging chunked files for sorting")
    output_multi_days_file = "player_top10_tmp.txt"
    merge_chunk_files(chunked_file_names, output_multi_days_file)
    tmp_join_output_dir = "tmp_join_output_dir"
    process_log_batches_with_workers(
        tmp_join_output_dir,
        worker_count,
        output_multi_days_file,
        batch_size,
        latest_date,
        worker_cache_multiplier,
    )
    # Cleanup
    delete_directory(tmp_all_files_folder_name)
    os.remove(output_multi_days_file)
    delete_directory(tmp_join_output_dir)
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
        default=100000,
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
    parser.add_argument(
        "--worker-cache-multiplier",
        type=int,
        default=4,
        help="Number of batches to read from logs for different workers to be consumed, low number could induce worker idle time, high number increases memory usage",
    )

    args = parser.parse_args()

    generate_statistics(
        args.files_to_consume,
        args.batch_size,
        args.workers,
        args.number_of_past_days,
        args.worker_cache_multiplier,
    )
