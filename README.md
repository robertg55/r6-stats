# R6 Siege Top 100 Operator Matches by Average Kills & Player Top 10 Matches by Kills - Log Processing Solution

## Overview

This solution processes Rainbow Six Siege match log files to compute the **Top 100 Operator Matches by Average Kills** and the **Top 10 Matches by Kills by Player**. Logs are sorted and processed in batches, with statistics updated daily. Multiprocessing is utilized to speed up computation by distributing the workload across multiple processes.

The primary goal is to generate daily statistics, computing the top 10 matches with the most kills for each player. If a player’s statistics are missing for the current day, the solution integrates the missing data from the previous day.

### Key Features

- **Batch Processing**: Processes logs in batches, optimizing memory usage and performance.
- **Sorting**: Log files are optionally sorted to ensure correct order.
- **Multiprocessing**: Utilizes multiple processes to handle large logs efficiently.
- **Daily Update**: Computes daily statistics while retaining and incorporating past data.
- **Handling Missing Data**: Retrieves missing player data from the previous day.
- **Queue-Based Parallelism**: Uses queues to manage work and results across processes.
- **Low memeory usage**: Uses file system to store data instead of memory.
- **Sample Log File Generation**: A dedicated script generates realistic test log files for custom scenarios.

## Directory Structure

├── logs/                                 # Folder containing application logs
├── player_top10_{date}/                  # Folder to store individual player statistics for the day
├── compute_operator.py                   # Python script to compute the top 100 Operators by average kills
├── compute_player.py                     # Python script to compute the top 10 matches by kills for all players
├── daily_operator_top100_{date}.txt      # Combined file for the top 100 Operators by average kills for a single day, reused if not deleted
├── log_sorter.py                         # Script to sort lines in a log file
├── log_util.py                           # Generic utility scripts used globally
├── match-sorted-r6-matches-{date}.txt    # r6-matches log file sorted by matches, reused if not deleted
├── operator_top100_{date}.txt            # Resulting file for the top 100 Operators by average kills for the last 7 days
├── player_top10_{date}.txt               # Resulting file for the top 10 matches by kills for all players
├── player-sorted-r6-matches-{date}.txt   # r6-matches log file sorted by player, reused if not deleted
├── r6-matches-{date}.log                 # r6-matches log input file
└── sample-log-file-generator.py          # Script for generating sample log data

## Prerequisites

Ensure you have the following installed:

- **Python 3.11+**
- **Pandas**: For data processing (`pip install pandas`)
- **NumPy**: For random number distributions (`pip install numpy`)

## Solution Overview

### Top 100 Operator Matches by Average Kills Solution

#### 1. Log Parsing with Optional Sorting

The solution processes logs from `r6-matches-YYYYMMDD.log` files. It extracts key match data, such as player IDs, operator IDs, match IDs, and kills, validating the structure of each log file. Logs should be sorted via the --sort param if they aren't already grouped by matches.

#### 2. Batch Processing

Logs are processed in batches, each containing a fixed number of log entries. This optimizes memory usage and ensures the data is processed efficiently. The batch doesn't cut in the middle of a match to properly compute averages by match.

#### 3. Top 100 Operators Calculation

The solution computes the top 100 operators based on the average number of kills per match. Results are stored to daily_operator_top100_{date}.txt for every date in the given computation range. Re-use existing files if a computed one already exists.

#### 4. Combining Results

After processing all previous days, the top 100 operator statistics from the previous given number of days are combined and saved to a file: `operator_top100_{date}.txt`.

### Player Top 10 Matches by Kills Solution

#### Step 1: Sorting the Logs

Logs are split into smaller chunks and sorted by player_id.

#### Step 2: Batch Processing

Once sorted, logs are processed in batches, The statistic for every player_id is computed and is stored to disk in a individual file inside folder called player_top10_{date}

#### Step 3: Combining Previous Days Data

Player data for the given amount of days is combined into a temporary folder with added date to avoid overwriting duplicate player data.

#### Step 4: Combining Results

Duplicate player data file is recomputed for top 10 and merged into a single file. Merged Outputs are stored in a separate folder where every player has a single file. A script then merges all files together into a single file player_top10_{date}.txt for the final results.

## How to Run the Solutions

### Daily Execution

To compute the top 100 operator matches by average kills OR the top 10 matches by kills for all players each day, follow these steps:

1. **Prepare the Log Files:**
   Ensure that the log files (named `r6-matches-YYYYMMDD.log`) are present in the current directory and the scripts also exist in the current directory. Each file should contain match statistics for a specific day.

2. **Set Python Path**:
   Set the `PYTHONPATH` environment variable to the current directory:

   - **Linux**:
     ```bash
     export PYTHONPATH=$(pwd):$PYTHONPATH
     ```
   - **Linux permanent python path**:
     ```bash
     echo 'export PYTHONPATH=$PYTHONPATH:$(pwd)' >> ~/.bashrc
     source ~/.bashrc
     ```
   - **Windows**:
     ```cmd
     set PYTHONPATH=%cd%;%PYTHONPATH%
     ```

3. **Run the Scripts**:

   - **For top 100 operator matches by average kills**:

     ```bash
     python compute_operator.py
     ```

   **Arguments:**
   - `--files-to-consume`: Comma-separated list of log files to process. If not provided, the script scan all available files.
   - `--batch-size`: Number of log entries to process per batch, larger batches increase memory usage and increase processing speed (default: 10,000).
   - `--workers`: Number of worker processes to run in parallel (default: 16).
   - `--number-of-past-days`: Number of past days to consider (default: 7).
   - `--sort`: Sorts the log files before processing if they are not already sorted by match number.

   - **For top 10 matches by kills for all players**:

     ```bash
     python compute_player.py
     ```

   **Arguments:**
   - `--files-to-consume`: Comma-separated list of log files to process. If not provided, the script scan all available files.
   - `--batch-size`: Number of log entries to process per batch, larger batches increase memory usage and increase processing speed (default: 10,000).
   - `--workers`: Number of worker processes to run in parallel (default: 16).
   - `--number-of-past-days`: Number of past days to consider (default: 7).

### Example Commands:

```bash
python compute_operator.py --batch-size 10000 --number-of-past-days 7 --workers 16 --sort
python compute_player.py --batch-size 10000 --number-of-past-days 7 --workers 16
```

### Automating Daily Runs

You can automate these scripts by adding them to a cron job (Linux/macOS) or Task Scheduler (Windows).

#### Example Cron Job (Linux/macOS):

```bash
0 0 * * * /usr/bin/python3 /path/to/scripts/compute_operator.py --batch-size 10000 --number-of-past-days 7 --workers 16 --sort
0 2 * * * /usr/bin/python3 /path/to/scripts/compute_player.py --batch-size 10000 --number-of-past-days 7 --workers 16
```

This runs the first script at midnight and the second at 2 AM every day.

## Output

- **Top 100 Operators by Kills**: Each day, results are saved to `operator_top100_{date}.txt`.
- **Player Statistics**: Each day, individual files for player statistics are stored in `player_top10_{date}/`.
- **Combined Player Statistics**: The file `player_top10_{date}.txt` stores the top 10 matches for all players.
- **Logs**: Logs include processed batches, skipped files, and completion messages.

## Logs

Logs track the progress of file processing, including:

- Files being processed.
- Batch sizes and the number of processed lines.
- Any skipped files or errors.

## Notes

- **File Naming**: Log files should follow the format `r6-matches-YYYYMMDD.log`.


## Sample Log File Generation

The `sample-log-file-generator.py` script generates random log files simulating R6 Siege matches. This script allows you to create test data with configurable parameters such as the number of players, matches, and dates.

### Features

- Generates realistic match logs with player and operator data.
- Can simulate 35 million players and 30 million matches per day.
- Allows smaller simulations for less players and matches to avoid long processing times.
- Supports intentional corruption of some lines (0.2%) for error testing.
- Supports custom date ranges for testing.

### Usage Example:

```bash
python sample-log-file-generator.py --players 1000 --matches 10000 --dates 20240901,20240902,20240903,20240904,20240905,20240906,20240907,20240908,20240909
```

This command generates logs with 1,000 unique players and 10,000 matches per day for the specified dates.


## Conclusion

This solution efficiently processes large volumes of Rainbow Six Siege match logs, computing both operator and player statistics in a scalable, memory-efficient manner. By leveraging multiprocessing, the solution is capable of handling large data sets quickly, making it ideal for daily processing.

For the top 10 matches by kills for all players, due to memory limitations, the solution currently relies on high I/O rates. To improve scalability and performance, especially for larger datasets, migrating to a thread-safe database (e.g., PostgreSQL or SQLite with WAL) would be recommended.
