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
- **Sample Log File Generation**: A dedicated script generates realistic test log files for custom scenarios.

## Directory Structure

```
.
├── chunks-{process_name}/             # Temporary folder to store chunk files during sorting
├── player_top10_{date}/               # Folder to store individual player statistics for the day
├── player_top10_{date}.txt            # Combined file for the top 10 matches by kills for all players
├── operator_top100_{date}.txt         # Combined file for the top 100 Operator matches by average kills
├── compute_operator.py                # Python script to compute the top 100 Operator matches by average kills
├── compute_player.py                  # Python script to compute the top 10 matches by kills for all players
├── log_sorter.py                      # Script to sort lines in a log file
├── log_util.py                        # Generic utility scripts used globally
└── sample-log-file-generator.py       # Script for generating sample log data
```

## Prerequisites

Ensure you have the following installed:

- **Python 3.11+**
- **Pandas**: For data processing (`pip install pandas`)
- **NumPy**: For random number distributions (`pip install numpy`)

## Solution Overview

### Top 100 Operator Matches by Average Kills Solution

#### 1. Log Parsing with Optional Sorting

The solution processes logs from `r6-matches-YYYYMMDD.log` files. It extracts key match data, such as player IDs, operator IDs, match IDs, and kills, validating the structure of each log file. Logs can be trimmed or sorted if they aren't already grouped by matches.

#### 2. Batch Processing

Logs are processed in batches, each containing a fixed number of log entries. This optimizes memory usage and ensures the data is processed efficiently.

#### 3. Top 100 Operators Calculation

The solution computes the top 100 operators based on the average number of kills per match. Results are aggregated across multiple batches and days, ensuring the most accurate statistics.

#### 4. Handling Missing Data

If operator data from the previous day is missing, the program retrieves it and integrates it with the current day's statistics, ensuring accuracy.

#### 5. Combining Results

After processing all batches, the top 100 operator statistics are combined and saved to a file: `operator_top100_{date}.txt`.

### Player Top 10 Matches by Kills Solution

#### Step 1: Sorting the Logs

Logs are split into smaller chunks and sorted in memory. Once sorted, the chunks are merged into a final sorted file for accurate processing.

#### Step 2: Batch Processing

Once sorted, logs are processed in batches, grouping each player’s match statistics. The top 10 matches (by kills) are computed for each player.

#### Step 3: Handling Missing Data

If a player’s data is missing for the current day, the solution retrieves the previous day’s data and integrates it.

#### Step 4: Combining Results

Once all player statistics are computed, they are combined into a single file, `player_top10_{date}.txt`, storing the top 10 matches by kills for each player.

### Sample Log File Generation

The `sample-log-file-generator.py` script generates random log files simulating R6 Siege matches. This script allows you to create test data with configurable parameters such as the number of players, matches, and dates.

#### Features

- Generates realistic match logs with player and operator data.
- Simulates up to 35 million players and 30 million matches per day.
- Supports intentional corruption of some lines (0.2%) for error testing.
- Supports custom date ranges for testing.

#### Usage Example:

```bash
python sample-log-file-generator.py --players 1000000 --matches 500000 --dates 20240901,20240902
```

This command generates logs with 1,000,000 unique players and 500,000 matches per day for the specified dates.

## How to Run the Solution

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

   - **Windows**:
     ```cmd
     set PYTHONPATH=%cd%;%PYTHONPATH%
     ```

3. **Run the Script**:

   - **For top 100 operator matches by average kills**:

     ```bash
     python compute_operator.py
     ```

   **Arguments:**
   - `--files-to-consume`: Comma-separated list of log files to process. If not provided, the script scan all available files.
   - `--batch-size`: Number of log entries to process per batch (default: 10,000).
   - `--workers`: Number of worker processes to run in parallel (default: 16).
   - `--number-of-past-days`: Number of past days to consider (default: 7).
   - `--sort`: Sorts the log files before processing if they are not already sorted by match number.

   - **For top 10 matches by kills for all players**:

     ```bash
     python compute_player.py
     ```

   **Arguments:**
   - `--files-to-consume`: Comma-separated list of log files to process. If not provided, the script scan all available files.
   - `--batch-size`: Number of log entries to process per batch (default: 10,000).
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

## Conclusion

This solution efficiently processes large volumes of Rainbow Six Siege match logs, computing both operator and player statistics in a scalable, memory-efficient manner. By leveraging multiprocessing, the solution is capable of handling large data sets quickly, making it ideal for daily processing.

For the top 10 matches by kills for all players, due to memory limitations, the solution currently relies on high I/O rates. To improve scalability and performance, especially for larger datasets, migrating to a thread-safe database (e.g., PostgreSQL or SQLite with WAL) would be recommended.
