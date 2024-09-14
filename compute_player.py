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


def computer_top_10_player_matches(df):
    """
    Computes the top 10 player matches by the total number of kills.

    This function groups the input DataFrame by `player_id` and `match_id`, summing the total number of kills (`nb_kills`)
    for each player in each match. It then retrieves the top 10 players based on their average number of kills using
    the `get_top_10_avg_kill_per_player_id` function.

    :param df: A DataFrame containing match statistics with columns such as `player_id`, `match_id`, and `nb_kills`.
    :type df: pandas.DataFrame

    :return: A DataFrame containing the top 10 players by their average number of kills per match.
    :rtype: pandas.DataFrame
    """
    grouped_df = df.groupby(["player_id", "match_id"])["nb_kills"].sum().reset_index()
    return get_top_10_avg_kill_per_player_id(grouped_df)


def get_top_10_avg_kill_per_player_id(df):
    """
    Retrieves the top 10 player matches by the number of kills.

    This function groups the input DataFrame by `player_id`, sorts each group by the `nb_kills` column in
    descending order, and selects the top 10 matches with the highest number of kills for each player.

    :param df: A DataFrame containing player statistics, including `player_id` and `nb_kills`.
    :type df: pandas.DataFrame

    :return: A DataFrame containing the top 10 matches for each player, sorted by the number of kills.
    :rtype: pandas.DataFrame
    """
    return (
        df.groupby("player_id")
        .apply(lambda group: group.sort_values(by="nb_kills", ascending=False).head(10))
        .reset_index(drop=True)
    )


def get_previous_day(current_date):
    """
    Calculates the previous day given a current date in 'YYYYMMDD' format.

    This function converts the `current_date` string to a `datetime` object, subtracts one day, and returns
    the previous day's date as a string in the same 'YYYYMMDD' format.

    :param current_date: The current date in 'YYYYMMDD' format.
    :type current_date: str

    :return: The previous day's date in 'YYYYMMDD' format.
    :rtype: str
    """
    date_obj = datetime.strptime(current_date, "%Y%m%d")
    previous_day = date_obj - timedelta(days=1)
    return previous_day.strftime("%Y%m%d")


def load_player_top_10_from_previous_day(current_date, files_in_directory):
    """
    Loads the top 10 player statistics from the previous day's file if it exists.

    This function attempts to load a file named `player_top10_<previous_day>.txt`, where `<previous_day>` is
    calculated based on the given `current_date`. If the file is found in the specified directory, it parses
    the file and returns the data as a DataFrame. Each line in the file is expected to follow the format:
    `player_id|match_id1:nb_kills1,match_id2:nb_kills2,...`. If the file is not found, the function returns `None`.

    :param current_date: The current date, used to compute the previous day's filename.
    :type current_date: str
    :param files_in_directory: A list of filenames in the current directory to check for the previous day's file.
    :type files_in_directory: list of str

    :return: A DataFrame containing the parsed statistics of the top 10 player matches from the previous day if the file exists, or `None` if the file is not found.
    :rtype: pandas.DataFrame or None
    """
    file_name = f"player_top10_{get_previous_day(current_date)}.txt"
    if file_name in files_in_directory:
        data = []
        with open(file_name, "r") as file:
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


def write_player_top_10_to_file(df, date):
    """
    Writes the top 10 player match statistics to a file named `player_top10_<date>.txt`.

    This function groups the input DataFrame by `player_id`, formats the matches and their associated kill counts,
    and writes the results to a text file. The output format for each line is:
    `player_id|match_id1:nb_kills1,match_id2:nb_kills2,...`.

    :param df: DataFrame containing player match statistics, with columns such as `player_id`, `match_id`, and `nb_kills`.
    :type df: pandas.DataFrame
    :param date: The date used to name the output file, in the format 'YYYYMMDD'.
    :type date: str

    :return: None
    """
    with open(f"player_top10_{date}.txt", "w") as file:
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


def concat_statistics_player_top_10(original_df, new_df):
    """
    Concatenates two DataFrames containing player statistics and returns the top 10 players by average kills.

    This function combines an existing DataFrame (`original_df`) with a new DataFrame (`new_df`). If the new DataFrame
    is `None`, it simply returns the original DataFrame. Otherwise, it concatenates the two DataFrames and computes
    the top 10 players by average kills using the `get_top_10_avg_kill_per_player_id` function.

    :param original_df: The original DataFrame containing player statistics.
    :type original_df: pandas.DataFrame
    :param new_df: The new DataFrame to concatenate with the original one. If None, only the original DataFrame is returned.
    :type new_df: pandas.DataFrame or None

    :return: A DataFrame containing the top 10 players by average kills after concatenating the two input DataFrames.
    :rtype: pandas.DataFrame
    """
    if new_df is None:
        return original_df
    return get_top_10_avg_kill_per_player_id(pd.concat([original_df, new_df]))


def get_files_in_current_directory():
    """
    Retrieves a list of all files in the current working directory.

    This function scans the current working directory and returns a list of files, excluding directories and
    other non-file items.

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
    Generates daily player statistics from R6 Siege match logs, focusing on the top 10 players by kills per match.

    This function processes log files from either a specified list (`files_to_consume`) or automatically detects
    relevant files in the current directory. It parses these files in batches, computes the top 10 players per match
    based on kills, and aggregates the statistics for each day. If the statistics for a day have already been computed,
    the file is skipped. The results are written to a file named `player_top10_<date>.txt`.

    :param files_to_consume: A comma-separated list of log files to process. If not provided, the function processes
                             all files starting with 'r6-matches-' in the current directory.
    :type files_to_consume: str
    :param batch_site: The number of rows to process in each batch for efficient log parsing.
    :type batch_site: int

    :return: None
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
        if f"player_top10_{date}.txt" in files_in_directory:
            # if it was already previously computed or the statistic generation is disabled
            print(f"already processed {file_path}, skipping")
            continue
        daily_player_top_10 = None

        for rows in parse_logs_from_file_in_batches(file_path, batch_site):
            # Create a DataFrame from the parsed log data
            df = pd.DataFrame(
                rows, columns=["player_id", "match_id", "operator_id", "nb_kills"]
            )

            # Convert 'operator_id' and 'nb_kills' columns to integers
            df["operator_id"] = df["operator_id"].astype(int)
            df["nb_kills"] = df["nb_kills"].astype(int)

            player_top_10 = computer_top_10_player_matches(df)
            daily_player_top_10 = concat_statistics_player_top_10(
                player_top_10, daily_player_top_10
            )
        previous_day_player_top_10 = load_player_top_10_from_previous_day(
            date, get_files_in_current_directory()
        )
        write_player_top_10_to_file(
            concat_statistics_player_top_10(
                daily_player_top_10, previous_day_player_top_10
            ),
            date,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computer statistics for R6 matches.")
    parser.add_argument(
        "--files-to-consume",
        type=str,
        help="Name of files to consume,comma separated, will process everything if not provided",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000000,
        help="Batch size for memory optimization",
    )

    args = parser.parse_args()

    generate_statistics(args.files_to_consume, args.batch_size)
