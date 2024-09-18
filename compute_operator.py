import pandas as pd
import uuid
import argparse
import os
import time
import multiprocessing
from log_sorter import sort_file
from log_util import (
    get_files_in_current_directory,
    setup_logger,
    signal_work_queue_finished,
)
from typing import Generator, Optional

logger = setup_logger()

WORKER_CACHE_MULTIPLIER = 4  # Number of log batches per worker to read in memory for workers to be ready consume, low number could induce worker idle time, high number increases memory usage


class LogMatchBatchParser:
    def __init__(self, file_path: str, batch_size: int = 100000, trimmed: bool = False):
        """
        Initializes the LogMatchBatchParser object.

        :param file_path: The path to the log file that will be parsed.
        :param batch_size: The number of lines to be processed in each batch.
        :param trimmed: Whether to add a fake player number for trimmed logs.
        """
        self.file_path = file_path
        self.batch_size = batch_size
        self.trimmed = trimmed
        self.max_kills = 45  # 9 max rounds for ranked with 1 players in each team killing all 5 enemies but going overtime for losing objective

    def parse_logs_from_file_in_batches(self) -> Generator[list[list[str]], None, None]:
        """
        Parses logs from the file in batches, yielding each batch of rows.

        The parsing does not split batches in the middle of a match statistic, ensuring
        the integrity of match data. It yields lists of rows, where each row is a list
        of values from the log file.

        :return: A generator that yields batches of log lines, where each batch contains
                 a list of lists, representing each line split by commas.
        """
        # Read the file contents
        last_match_id = None
        split_batch_trigger = False
        start_time = time.time()
        with open(self.file_path, "r") as file:
            batch = []
            for line_number, line in enumerate(file, start=1):
                if line_number % 1000000 == 0:
                    logger.info(
                        f"processing {self.file_path} line number {line_number} with elapsed seconds {time.time()-start_time}"
                    )
                if self.is_valid_row(line):
                    # Split each line by commas to create batch
                    new_line = line.strip().split(",")
                    if self.trimmed:
                        new_line = [
                            "0"
                        ] + new_line  # Adds fake player number to avoid different columns number logic
                    if last_match_id is None:
                        # Handles first row
                        last_match_id = new_line[1]
                    if (
                        split_batch_trigger
                        and last_match_id != new_line[1]
                        and batch != []
                    ):
                        # Doesn't split the batch in the middle of a match statistic
                        split_batch_trigger = False
                        last_match_id = new_line[1]
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
        Validates if the given row has the correct format and data types.

        :param row: A comma-separated string representing a log row.
        :return: True if the row is valid, False otherwise.
        """
        try:
            if not self.trimmed:
                player_id, match_id, operator_id, nb_kills = row.split(",")
                uuid.UUID(player_id)  # Validate UUID
            else:
                match_id, operator_id, nb_kills = row.split(",")
            uuid.UUID(match_id)  # Validate UUID
            int(operator_id)  # Validate operator_id as integer
            kills = int(nb_kills)  # Validate nb_kills as integer
            if kills > self.max_kills:
                return False
            return True
        except Exception:
            return False


def compute_top_100_average_number_kill_per_operator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the top 100 operators by average number of kills per match.

    This function groups the input DataFrame by `operator_id` and `match_id`, calculating the average number
    of kills (`avg_kills`) for each operator in each match. It then retrieves the top 100 operators by
    average kills using the `get_top_100_avg_kill_per_operator_id` function.

    :param df: DataFrame containing operator statistics, with columns such as `operator_id`, `match_id`, and `nb_kills`.

    :return: A DataFrame containing the top 100 operators by average kills per match.
    """
    # Group by 'operator_id' and 'match_id' to compute average kills per operator per match
    grouped_df = (
        df.groupby(["operator_id", "match_id"])
        .agg(avg_kills=("nb_kills", "mean"))
        .reset_index()
    )
    return get_top_100_avg_kill_per_operator_id(grouped_df)


def get_top_100_avg_kill_per_operator_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieves the top 100 operators by average kills from a given DataFrame.

    This function groups the input DataFrame by `operator_id`, then sorts each group by the `avg_kills` column
    in descending order. For each operator, it selects the top 100 entries and returns the result as a new DataFrame.

    :param df: DataFrame containing the operator statistics, including `operator_id` and `avg_kills` columns.
    :return: A DataFrame containing the top 100 entries for each operator, sorted by average kills in descending order.
    """
    return (
        df.groupby("operator_id")
        .apply(
            lambda group: group.sort_values(by="avg_kills", ascending=False).head(100)
        )
        .reset_index(drop=True)
    )


def load_operator_top_100_from_specific_day(file_name: str) -> pd.DataFrame:
    """
    Loads the top 100 operators' statistics from a specific day.

    This function reads the statistics from a file and converts the data into a DataFrame.

    :param file_name: The path to the file containing operator statistics for a specific day.
    :return: A DataFrame containing the operator statistics or None if the file is empty.
    """

    data = []
    with open(file_name, "r") as file:
        for line in file:
            operator_id, matches = line.strip().split("|")

            # Split the matches and avg_kills part
            match_avg_kills = matches.split(",")

            for match_avg in match_avg_kills:
                match_id, avg_kills = match_avg.split(":")

                # Append the parsed data to the list
                data.append(
                    {
                        "operator_id": int(operator_id),
                        "match_id": match_id,
                        "avg_kills": float(avg_kills),
                    }
                )

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    return df


def write_operator_top_100_to_file(df: pd.DataFrame, filename: str) -> None:
    """
    Writes the top 100 operator statistics to a file.

    This function converts the DataFrame containing operator statistics into a formatted string and writes it to a file.

    :param df: A DataFrame containing the top 100 operator statistics.
    :param filename: The path to the file where the statistics will be written.
    """
    with open(filename, "w") as file:
        # Output format: operator_id|match_id1:avg_kills1,match_id2:avg_kills2,...
        output = df.groupby("operator_id").apply(
            lambda group: f"{group['operator_id'].iloc[0]}|"
            + ",".join(
                [
                    f"{row['match_id']}:{row['avg_kills']:.2f}"
                    for _, row in group.iterrows()
                ]
            )
        )
        for line in output:
            file.write(line + "\n")


def concat_statistics_operator_top_100(
    original_df: pd.DataFrame, new_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Concatenates two DataFrames containing operator statistics and returns the top 100 operators by average kills.

    This function combines an existing DataFrame (`original_df`) with a new DataFrame (`new_df`). If the new DataFrame
    is `None`, it simply returns the original DataFrame. Otherwise, it concatenates the two DataFrames and computes
    the top 100 operators by average kills.

    :param original_df: The original DataFrame containing operator statistics.
    :param new_df: The new DataFrame to concatenate with the original one. If None, only the original DataFrame is returned.

    :return: A DataFrame containing the top 100 operators by average kills after concatenating the two input DataFrames.
    """
    if new_df is None:
        return original_df
    return get_top_100_avg_kill_per_operator_id(pd.concat([original_df, new_df]))


def process_batch(batch: list[list[str]]) -> dict[int, float]:
    """
    Processes a batch of log data by creating a DataFrame and computing the top 100
    operators by average number of kills.

    :param batch: A list of lists where each inner list represents a row from the log
                  file with columns ['player_id', 'match_id', 'operator_id', 'nb_kills'].
    :return: A dictionary where the keys are operator IDs and the values are the
             average number of kills for the top 100 operators.
    """
    # Create a DataFrame from the parsed log data
    df = pd.DataFrame(
        batch, columns=["player_id", "match_id", "operator_id", "nb_kills"]
    )

    # Convert 'operator_id' and 'nb_kills' columns to integers
    df["operator_id"] = df["operator_id"].astype(int)
    df["nb_kills"] = df["nb_kills"].astype(int)

    operator_top_100 = compute_top_100_average_number_kill_per_operator(df)
    return operator_top_100


def worker(
    work_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue
) -> None:
    """
    Worker function that processes batches of tasks from the work queue and aggregates results.

    This function continuously retrieves tasks from the `work_queue`, processes each batch, and
    accumulates the top 100 operators' statistics. Once all tasks are processed (signaled by
    a `None` task in the queue), the aggregated result is placed in the `result_queue`.

    The worker keeps running until it encounters a `None` in the `work_queue`, which indicates
    the end of tasks. At that point, it places the accumulated results into the `result_queue`
    and signals that it has finished processing.

    :param work_queue: A multiprocessing.Queue object containing batches of tasks to be processed.
    :param result_queue: A multiprocessing.Queue object where the aggregated results are placed.
    """
    daily_operator_top_100 = None
    while True:
        task = work_queue.get()
        if task is None:
            result_queue.put(daily_operator_top_100)

        result = process_batch(task)
        daily_operator_top_100 = concat_statistics_operator_top_100(
            result, daily_operator_top_100
        )
    signal_work_queue_finished(work_queue)


def multi_process_batch_consumption(
    file_to_consume: str,
    worker_count: int,
    batch_size: int,
    requires_sorting: bool,
    date: str,
    worker_cache_multiplier: int,
) -> None:
    """
    Processes a log file using multiple workers and updates the top 100 operators' statistics.

    This function uses multiprocessing to parse and process a log file in batches. Each batch is consumed by a pool of
    worker processes, which compute statistics and aggregate the top 100 operators by kills. The results are gathered
    from all workers and combined into a final set of statistics, which is then written to an output file named with the
    provided `date`.

    The function supports both sorted and unsorted files, handles batch consumption, and ensures proper cleanup of
    processes and temporary files once the task is complete.

    :param file_to_consume: The path to the log file that will be processed.
    :param worker_count: The number of worker processes to use for parallel processing.
    :param batch_size: The number of log entries to process per batch.
    :param requires_sorting: Boolean flag indicating whether the file was sorted before processing.
    :param date: The date string used for naming the output file.
    :param worker_cache_multiplier: Multiplier for worker queue caching, determining how many batches to queue for each worker.
    """
    # Create a multiprocessing Queue to store results in a thread-safe way
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    work_queue = manager.Queue()
    workers = []
    for _ in range(worker_count):
        p = multiprocessing.Process(target=worker, args=(work_queue, result_queue))
        workers.append(p)
        p.start()
    parser = LogMatchBatchParser(
        file_to_consume, batch_size=batch_size, trimmed=requires_sorting
    )
    log_parser_generator = parser.parse_logs_from_file_in_batches()

    next_batch = None
    while True:
        if work_queue.qsize() < worker_count * worker_cache_multiplier:
            next_batch = next(log_parser_generator, None)
            work_queue.put(next_batch)
            if next_batch is None:
                signal_work_queue_finished(work_queue)
                break
    daily_operator_top_100 = None
    for worker_process in workers:
        results_df = result_queue.get()  # Waits on output results from every worker
        daily_operator_top_100 = concat_statistics_operator_top_100(
            results_df, daily_operator_top_100
        )
    write_operator_top_100_to_file(
        daily_operator_top_100, f"daily_operator_top100_{date}.txt"
    )
    # Cleanup
    for worker_process in workers:
        worker_process.terminate()
    manager.shutdown()


def generate_daily_operator_top100(
    files_to_consume: Optional[str],
    number_of_past_days: int,
    requires_sorting: bool,
    worker_count: int,
    batch_size: int,
    worker_cache_multiplier: int,
) -> str:
    """
    Processes log files and generates the top 100 operators' statistics for the past `number_of_past_days`.

    This function processes R6 Siege match logs to compute the top 100 operators by average kills. It processes the
    files from the past `number_of_past_days`, optionally sorts them by operator, and then performs batch consumption
    using multiple workers. If a file has already been processed, it will be skipped.

    The function returns the latest date processed, which is used in naming the final output file.

    :param files_to_consume: A comma-separated string of filenames to process, or None to process all matching files
                             from the directory.
    :param number_of_past_days: The number of past days' worth of log files to include in the analysis.
    :param requires_sorting: Boolean flag indicating whether the files need to be sorted by operator before processing.
    :param worker_count: The number of worker processes to use for parallel processing.
    :param batch_size: The number of log entries to process per batch.
    :param worker_cache_multiplier: Multiplier for worker queue caching, determining how many batches to queue for each worker.
    :return: A string representing the latest date that was processed for which statistics will be generated.
    """
    files_in_directory = get_files_in_current_directory()
    if not files_to_consume:
        r6_files = [
            file for file in files_in_directory if file.startswith("r6-matches-")
        ]
    else:
        r6_files = files_to_consume.split(",")
    sorted_r6_files = sorted(r6_files)[-number_of_past_days:]
    logger.info(
        f"processing operator top 100 matches by kill average for files {sorted_r6_files}"
    )

    date = None
    for file_path in sorted_r6_files:
        date = file_path[11:-4]
        if f"daily_operator_top100_{date}.txt" in files_in_directory:
            # if it was already previously computed or the statistic generation is disabled
            logger.info(f"already processed {file_path}, skipping")
            continue
        if requires_sorting:
            new_file_path = f"match-sorted-{file_path}"
            sort_file(file_path, new_file_path, "operator", True)
        else:
            new_file_path = file_path
        multi_process_batch_consumption(
            new_file_path,
            worker_count,
            batch_size,
            requires_sorting,
            date,
            worker_cache_multiplier,
        )
        if requires_sorting:
            # cleanup of sorted files
            os.remove(new_file_path)
    return date


def generate_statistics(
    files_to_consume: Optional[str],
    batch_size: int,
    worker_count: int,
    number_of_past_days: int,
    requires_sorting: bool,
    worker_cache_multiplier: int,
) -> None:
    """
    Generates the top 100 operator statistics from log files for the specified days.

    This function processes log files in batches, computes the top 100 operators by average kills, and writes
    the results to files. It uses multiprocessing to parallelize the task.

    :param files_to_consume: A comma-separated string of filenames to process, or None to process all available files.
    :param batch_size: The number of log entries to process per batch.
    :param worker_count: The number of worker processes to use for parallel processing.
    :param number_of_past_days: The number of past days to include in the analysis.
    :param requires_sorting: A flag indicating whether the log files need to be sorted before processing.
    :param worker_cache_multiplier: Multiplier for worker queue caching, determining how many batches to queue for each worker.
    """
    latest_date = generate_daily_operator_top100(
        files_to_consume,
        number_of_past_days,
        requires_sorting,
        worker_count,
        batch_size,
        worker_cache_multiplier,
    )

    files_in_directory = get_files_in_current_directory()
    daily_files = [
        file for file in files_in_directory if file.startswith("daily_operator_top100_")
    ]
    sorted_daily_files = sorted(daily_files)[-number_of_past_days:]
    daily_operator_top_100 = None
    for daily_file in sorted_daily_files:
        # Combines previous single day files into one
        previous_day_operator_top_100 = load_operator_top_100_from_specific_day(
            daily_file
        )
        daily_operator_top_100 = concat_statistics_operator_top_100(
            daily_operator_top_100, previous_day_operator_top_100
        )

    write_operator_top_100_to_file(
        daily_operator_top_100, f"operator_top100_{latest_date}.txt"
    )
    logger.info("finished computing operator top 100 matches by kill average")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computer operator top 100 matches by kill average for R6 matches."
    )
    parser.add_argument(
        "--files-to-consume",
        type=str,
        help="Name of files to consume, comma separated, will process everything if not provided",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for memory optimization",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of separate workers running in parallel",
    )
    parser.add_argument(
        "--number-of-past-days",
        type=int,
        default=7,
        help="Only consumes this many days in the past",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort the log files if they aren't already sorted by match number",
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
        args.sort,
        args.worker_cache_multiplier,
    )
