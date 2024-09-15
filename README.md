# R6 Siege Top 100 Operator Matches by Average Kills & Player Top 10 Matches by Kills - Log Processing Solution

## Overview

This solution processes Rainbow Six Siege match log files to compute the top 100 Operator Matches by Average Kills and top 10 matches by kills for each player. The logs are sorted and processed in batches, and the statistics are updated each day. The solution uses multiprocessing to speed up computation by splitting the workload across multiple processes.

The primary goal is to generate daily statistics, computing the top 10 matches with the most kills for each player. If a player’s statistics are missing for the current day, the solution adds the missing statistics from the previous day.

### Key Features

- **Batch Processing:** The solution processes logs in batches, optimizing memory usage and performance.
- **Sorting:** Log files are sorted before processing to ensure the correct order of data.
- **Multiprocessing:** Multiple processes are used to speed up the computation, especially for large logs.
- **Daily Update:** Each day, the solution computes new statistics while retaining past data.
- **Handling Missing Data:** If statistics are missing for a player, they are retrieved from the previous day.
- **Queue-Based Parallelism:** Work and result queues manage tasks and results between worker processes.

## Directory Structure

```
.
├── chunks-{process_name}/     # Temporary folder to store chunk files during sorting
├── player_top10_{date}/       # Folder to store individual player statistics for the given day
├── player_top10_{date}.txt    # Final combined file for the top 10 matches by kills for all players
├── operator_top100_{date}.txt    # Final combined file for the top 100 Operator matches by average kills
├── compute_operator.py    #  Python script to compute the top 100 Operator matches by average kills
├── compute_player.py   # Python script to compute top 10 matches by kills for all players
├── log_sorter.py    # Python Script to sort lines in a log file
├── log_util.py    # Generic python scripts used globally
└── sample-log-file-generator.py                   # Python scripts for generating sample data
```

## Prerequisites

Make sure you have the following installed:

- **Python 3.11+**
- **Pandas**: For data processing (`pip install pandas`)

## Solution Overview


### Top 100 Operator Matches by Average Kills Solution

#### 1. Log Parsing with optional sorting

The solution processes logs from `r6-matches-YYYYMMDD.log` files. It extracts key match data, such as player IDs, operator IDs, match IDs, and kills, and validates the structure of the log files. The log files can be trimmed or sorted as needed if they aren't already sorted/grouped by matches.

#### 2. Batch Processing

Logs are processed in batches, where each batch contains a fixed number of log entries. This helps optimize memory usage and ensures that the data is processed in manageable chunks.

#### 3. Top 100 Operators Calculation

The program computes the top 100 operators based on the average number of kills per match. The results are concatenated across multiple batches and days, ensuring that the most up-to-date statistics are available.

#### 4. Handling Missing Data

If an operator’s data from the previous day is missing, the program retrieves and integrates it with the current day’s statistics, ensuring complete and accurate results.

#### 5. Combining Results

Once all batches are processed, the top 100 operator statistics for each day are combined and written to a file: `operator_top100_{date}.txt`.

### Player Top 10 Matches by Kills

#### Step 1: Sorting the Logs

Log files are split into smaller chunks and sorted in memory. Once sorted, the chunks are merged into a final sorted file. This ensures that the logs are in the correct order for processing.

#### Step 2: Batch Processing

Once sorted, logs are processed in batches. Each player’s match statistics are grouped, and the top 10 matches (by kills) are computed for each player.

#### Step 3: Handling Missing Data

If a player’s data is missing for the current day, the solution checks the previous day’s data and adds it to the current day’s records.

#### Step 4: Combining Results

Once all individual player statistics are computed, they are combined into a single file that stores the top 10 matches for each player by kills.

### Sample Log File Generation

The `sample-log-file-generator.py` script generates random log files simulating game scenarios with player and operator statistics. It allows you to create realistic test data with configurable parameters like the number of players, matches, and dates.

#### Features of `sample-log-file-generator.py`

- Generates log files simulating R6 Siege matches with realistic player and operator data.
- Simulates up to 35 million players and 30 million matches per day.
- Allows intentional corruption of some lines (0.2% chance) to simulate real-world errors.
- Supports custom date ranges and multiple dates for testing.
  
#### Usage Example:

```bash
python sample-log-file-generator.py --players 1000000 --matches 500000 --dates 20240901,20240902
```

This command will generate log files with 1,000,000 unique players and 500,000 matches per day for the specified dates.

## How to Run the Solution

### Daily Execution

To compute the top 100 operator matches by average kills OR the top 10 matches by kills for all players each day, follow these steps:

1. **Prepare the Log Files:**
   Ensure that the log files (named `r6-matches-YYYYMMDD.log`) are present in the current directory and the scripts also exist in the current directory. Each file should contain match statistics for a specific day.

2. **Set python path**
   Set python path environment variable to current path to ensure that the scripts files can find each other
   Linux:
```bash
# Linux:
export PYTHONPATH=$(pwd):$PYTHONPATH
```
   Windows:
```cmd
:: Windows:
set PYTHONPATH=%cd%;%PYTHONPATH%
```

3. **Run the Script:**

   Use the following command to run the script for daily log processing:

   For top 100 operator matches by average kills:

   ```bash
   python compute_operator.py 
   ```

   **Arguments:**
   - `--files-to-consume`: Comma-separated list of log files to process. If not provided, the script scan all available files.
   - `--batch-size`: Number of log entries to process per batch (default: 10,000).
   - `--workers`: Number of worker processes to run in parallel (default: 16).
   - `--number-of-past-days`: Number of past days to consider (default: 7).
   - `--sort`: Sorts the log files before processing if they are not already sorted by match number.

   For top 10 matches by kills for all players each day:

   ```bash
   python compute_player.py
   ```

   - `--files-to-consume`: Comma-separated list of log files to process. If not provided, the script scan all available files.
   - `--batch-size`: Number of log entries to process per batch (default: 10,000).
   - `--workers`: Number of worker processes to run in parallel (default: 16).
   - `--number-of-past-days`: Number of past days to consider (default: 7).
   

### Example Commands

```bash
python compute_operator.py --batch-size 10000 --number-of-past-days 7 --workers 16 --sort
python compute_player.py --batch-size 10000 --number-of-past-days 7 --workers 16
```

This command will process the log file the last 7 days it finds, using 16 worker processes, with a batch size of 10,000 entries. In the case of the top 100 operators, it will sort the file.

### Automating Daily Runs

You can automate the script to run daily by adding a cron job (Linux/macOS) or Task Scheduler (Windows) to execute the script every day.

#### Example Cron Job (Linux/macOS)

To run the script every day at midnight, add the following line to your crontab:

```bash
0 0 * * * /usr/bin/python3 /path/to/scripts/compute_operator.py --batch-size 10000 --number-of-past-days 7 --workers 16 --sort
0 2 * * * /usr/bin/python3 /path/to/scripts/compute_player.py --batch-size 10000 --number-of-past-days 7 --workers 16
```

This will execute the first script every day at 00:00 and the second at 02:00.

## Output
- **Top 100 Operators by Kills**: Each day, the top 100 operators by average kills will be written to `operator_top100_{date}.txt`.
- **Individual Player Statistics**: For each day, individual files containing the top 10 matches for each player are stored in the `player_top10_{date}` folder.
- **Combined Player Statistics**: The script generates a combined file, `player_top10_{date}.txt`, which stores the top 10 matches for all players.
- **Logs**: The script logs progress, including the number of processed batches, skipped files, and any errors.

## Logs

The solution logs the progress of the file processing. Logs are output to the console, including the following information:

- Files being processed.
- Batch sizes and the number of processed lines.
- Any skipped files (if they have already been processed).
- Completion messages.

## Notes

- **File Naming**: Log files should follow the format `r6-matches-YYYYMMDD.log` for proper processing.
- **Chunk Size**: The chunk size is set to 512 MB by default. You can adjust this if you're working with larger or smaller files.

## Conclusion

This solution efficiently processes large volumes of Rainbow Six Siege match logs, computing both operator and player statistics in a scalable, memory-efficient manner. By leveraging multiprocessing, the solution is capable of handling large data sets quickly, making it ideal for daily processing.

For the top 10 matches by kills for all players, due to memory limitations, the solution currently relies on high I/O rates. To improve scalability and performance, especially for larger datasets, migrating to a thread-safe database (e.g., PostgreSQL or SQLite with WAL) would be recommended.