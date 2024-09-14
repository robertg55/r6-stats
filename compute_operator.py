import pandas as pd
import uuid
import argparse
import os
from datetime import datetime, timedelta


def is_valid_row(row):
    """
    Validates a log entry to ensure it contains the correct format and data types.

    This function checks if a row (log entry) contains four fields: `player_id`, `match_id`, `operator_id`,
    and `nb_kills`. It validates that `player_id` and `match_id` are valid UUIDs, and that `operator_id` and
    `nb_kills` are integers. If the row meets these criteria, it returns `True`; otherwise, it returns `False`.

    :param row: A string representing a log entry, expected to contain four comma-separated values: `player_id`,
                `match_id`, `operator_id`, and `nb_kills`.
    :type row: str

    :return: `True` if the row is valid, otherwise `False`.
    :rtype: bool
    """
    try:
        player_id, match_id, operator_id, nb_kills = row.split(",")
        uuid.UUID(player_id)  # Validate UUID
        uuid.UUID(match_id)  # Validate UUID
        int(operator_id)  # Validate operator_id as integer
        int(nb_kills)  # Validate nb_kills as integer
        return True
    except ValueError:
        return False


def parse_logs_from_file_in_batches(file_path, batch_size=100000):
    """
    Parses a log file in batches, yielding each batch of rows while ensuring match statistics are not split.

    This function reads a file line by line, creating batches of log entries of the specified size. It ensures
    that the batch is not split in the middle of match statistics by checking the `match_id` of each row. When a batch
    reaches the specified `batch_size`, it is yielded. The function continues until all lines in the file are processed.

    :param file_path: The path to the log file that needs to be parsed.
    :type file_path: str
    :param batch_size: The number of lines to include in each batch before yielding. Default is 100,000.
    :type batch_size: int, optional

    :yield: A list of rows (each row as a list of fields) for each batch of log entries.
    :rtype: generator of list of list of str

    :raises FileNotFoundError: If the file specified by `file_path` does not exist.

    :note: The function ensures that rows from the same match are not split across different batches,
           using `match_id` as a reference.
    """
    # Read the file contents
    last_match_id = None
    split_batch_trigger = False

    batch_number = 0
    with open(file_path, "r") as file:
        batch = []
        for line_number, line in enumerate(file, start=1):
            if is_valid_row(line):
                # Split each line by commas to create batch
                new_line = line.strip().split(",")
                batch.append(new_line)  # Add line to batch

            if line_number % batch_size == 0:
                split_batch_trigger = True
            if split_batch_trigger and last_match_id != new_line[1]:
                # Doesn't split the batch in the middle of a match statistic
                split_batch_trigger = False
                print(f"processing {file_path} batch number {batch_number}")
                batch_number += 1
                yield batch  # Yield the batch when batch_size is reached
                batch = []  # Reset the batch
            last_match_id = new_line[1]
        if batch:
            yield batch  # Yield the remaining lines if there are any


def compute_top_100_average_numper_kill_per_operator(df):
    """
    Computes the top 100 operators by average number of kills per match.

    This function groups the input DataFrame by `operator_id` and `match_id`, calculating the average number
    of kills (`avg_kills`) for each operator in each match. It then retrieves the top 100 operators by
    average kills using the `get_top_100_avg_kill_per_operator_id` function.

    :param df: DataFrame containing operator statistics, with columns such as `operator_id`, `match_id`, and `nb_kills`.
    :type df: pandas.DataFrame

    :return: A DataFrame containing the top 100 operators by average kills per match.
    :rtype: pandas.DataFrame
    """
    # Group by 'operator_id' and 'match_id' to compute average kills per operator per match
    grouped_df = (
        df.groupby(["operator_id", "match_id"])
        .agg(avg_kills=("nb_kills", "mean"))
        .reset_index()
    )
    return get_top_100_avg_kill_per_operator_id(grouped_df)


def get_top_100_avg_kill_per_operator_id(df):
    """
    Retrieves the top 100 operators by average kills from a given DataFrame.

    This function groups the input DataFrame by `operator_id`, then sorts each group by the `avg_kills` column
    in descending order. For each operator, it selects the top 100 entries and returns the result as a new DataFrame.

    :param df: DataFrame containing the operator statistics, including `operator_id` and `avg_kills` columns.
    :type df: pandas.DataFrame

    :return: A DataFrame containing the top 100 entries for each operator, sorted by average kills in descending order.
    :rtype: pandas.DataFrame
    """
    return (
        df.groupby("operator_id")
        .apply(
            lambda group: group.sort_values(by="avg_kills", ascending=False).head(100)
        )
        .reset_index(drop=True)
    )


def get_previous_day(current_date):
    """
    Calculates the previous day's date given a current date in 'YYYYMMDD' format.

    This function takes a date string in the format 'YYYYMMDD', converts it to a `datetime` object,
    subtracts one day, and returns the previous day as a string in the same 'YYYYMMDD' format.

    :param current_date: The current date as a string in 'YYYYMMDD' format.
    :type current_date: str

    :return: The previous day's date as a string in 'YYYYMMDD' format.
    :rtype: str
    """
    date_obj = datetime.strptime(current_date, "%Y%m%d")
    previous_day = date_obj - timedelta(days=1)
    return previous_day.strftime("%Y%m%d")


def load_operator_top_100_from_previous_day(current_date, files_in_directory):
    """
    Loads the top 100 operator statistics from the previous day's file if it exists.

    This function attempts to load a file named `operator_top100_<previous_day>.txt`, where `<previous_day>` is
    computed based on the given `current_date`. If the file is found in the specified directory, it parses the
    file and returns the data as a DataFrame. Each line in the file is expected to follow the format:
    `operator_id|match_id1:avg_kills1,match_id2:avg_kills2,...`. If the file does not exist, the function returns `None`.

    :param current_date: The current date, used to compute the previous day's filename.
    :type current_date: str
    :param files_in_directory: A list of filenames in the current directory to check for the previous day's file.
    :type files_in_directory: list of str

    :return: A DataFrame containing the parsed statistics from the previous day if the file exists, or `None` if the file is not found.
    :rtype: pandas.DataFrame or None
    """
    file_name = f"operator_top100_{get_previous_day(current_date)}.txt"
    if file_name in files_in_directory:
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
    return None


def write_operator_top_100_to_file(df, date):
    """
    Writes the top 100 operator statistics to a file named `operator_top100_<date>.txt`.

    This function takes a DataFrame containing operator statistics and writes the data to a text file. The output
    format for each line is: `operator_id|match_id1:avg_kills1,match_id2:avg_kills2,...`, where each `operator_id`
    is followed by a list of matches and their respective average kills.

    :param df: DataFrame containing the top 100 operators' statistics, with columns such as `operator_id`, `match_id`,
               and `avg_kills`.
    :type df: pandas.DataFrame
    :param date: The date used to name the output file, in the format 'YYYY-MM-DD'.
    :type date: str

    :return: None
    """
    with open(f"operator_top100_{date}.txt", "w") as file:
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


def concat_statistics_operator_top_100(original_df, new_df):
    """
    Concatenates two DataFrames containing operator statistics and returns the top 100 operators by average kills.

    This function combines an existing DataFrame (`original_df`) with a new DataFrame (`new_df`). If the new DataFrame
    is `None`, it simply returns the original DataFrame. Otherwise, it concatenates the two DataFrames and computes
    the top 100 operators by average kills.

    :param original_df: The original DataFrame containing operator statistics.
    :type original_df: pandas.DataFrame
    :param new_df: The new DataFrame to concatenate with the original one. If None, only the original DataFrame is returned.
    :type new_df: pandas.DataFrame or None

    :return: A DataFrame containing the top 100 operators by average kills after concatenating the two input DataFrames.
    :rtype: pandas.DataFrame
    """
    if new_df is None:
        return original_df
    return get_top_100_avg_kill_per_operator_id(pd.concat([original_df, new_df]))


def get_files_in_current_directory() -> list[str]:
    """
    Retrieves a list of all files in the current working directory.

    This function scans the current directory and returns only the files, excluding directories and other
    non-file items.

    :return: A list of filenames found in the current working directory.
    :rtype: list of str
    """
    current_directory = os.getcwd()
    return [
        file
        for file in os.listdir(current_directory)
        if os.path.isfile(os.path.join(current_directory, file))
    ]


def generate_statistics(files_to_consume, batch_site):
    """
    Generates daily statistics from R6 Siege match logs, focusing on the top 100 operators by average kills.

    This function processes log files either from a provided list (`files_to_consume`) or automatically detects
    relevant files in the current directory. It parses these files in batches, computes the average number
    of kills per operator, and aggregates the top 100 operators for each day. If the statistics for a day
    have already been computed, the file is skipped. The results are written to a file named 'operator_top100_<date>.txt'.

    :param files_to_consume: A comma-separated list of log files to process. If empty, the function will
                             process all files in the current directory that start with 'r6-matches-'.
    :type files_to_consume: str
    :param batch_site: The size of the batch for reading log files in chunks, allowing efficient processing.
    :type batch_site: int

    :return: None

    :raises FileNotFoundError: If the log files are not found in the specified directory.
    :raises ValueError: If the log files contain invalid data or if the batch processing fails.

    :note:
        - The function merges the computed statistics with data from the previous day, if available.
        - Daily statistics are written to a file with the format 'operator_top100_<date>.txt'.
    """
    files_in_directory = get_files_in_current_directory()
    if not files_to_consume:
        r6_files = [
            file for file in files_in_directory if file.startswith("r6-matches-")
        ]
    else:
        r6_files = files_to_consume.split(",")
    sorted_r6_files = sorted(r6_files)
    print(f"processing files {sorted_r6_files}")
    for file_path in sorted_r6_files:
        date = file_path[11:-4]
        if f"operator_top100_{date}.txt" in files_in_directory:
            # if it was already previously computed or the statistic generation is disabled
            print(f"already processed {file_path}, skipping")
            continue
        daily_operator_top_100 = None

        for rows in parse_logs_from_file_in_batches(file_path, batch_site):
            # Create a DataFrame from the parsed log data
            df = pd.DataFrame(
                rows, columns=["player_id", "match_id", "operator_id", "nb_kills"]
            )

            # Convert 'operator_id' and 'nb_kills' columns to integers
            df["operator_id"] = df["operator_id"].astype(int)
            df["nb_kills"] = df["nb_kills"].astype(int)

            operator_top_100 = compute_top_100_average_numper_kill_per_operator(df)
            daily_operator_top_100 = concat_statistics_operator_top_100(
                operator_top_100, daily_operator_top_100
            )

        previous_day_operator_top_100 = load_operator_top_100_from_previous_day(
            date, get_files_in_current_directory()
        )
        write_operator_top_100_to_file(
            concat_statistics_operator_top_100(
                daily_operator_top_100, previous_day_operator_top_100
            ),
            date,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computer statistics for R6 matches.")
    parser.add_argument(
        "--files-to-consume",
        type=str,
        help="Name of files to consume, comma separated, will process everything if not provided",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000000,
        help="Batch size for memory optimization",
    )

    args = parser.parse_args()

    generate_statistics(args.files_to_consume, args.batch_size)
