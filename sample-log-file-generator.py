import uuid
import argparse
import numpy as np
import random
import time
from datetime import datetime
from log_util import setup_logger

logger = setup_logger()


def generate_log_file(num_players: int, num_matches: int, dates: str) -> None:
    """
    Generates log files for R6 Siege matches with player and operator statistics, simulating various game scenarios.

    This function creates sample log files containing data for a specified number of players and matches, distributed across multiple dates. Each match includes unique player and operator data, and the total number of kills is distributed among the players. A percentage of log entries may be intentionally corrupted to simulate errors.

    :param num_players: The total number of unique players for whom the log data will be generated.
    :type num_players: int
    :param num_matches: The total number of matches to simulate in each log file.
    :type num_matches: int
    :param dates: A comma-separated string of dates for which to generate log files (e.g., '20230101,20230102').
    :type dates: str

    :return: None

    :note: The log file format is as follows for each valid line:
           `player_id,match_id,operator_id,nb_kills`. Corrupted lines may have missing or incorrectly ordered fields.

    :raises AssertionError: If the length of `NUMBER_OF_PICKED_DIFFERENT_OPERATOR_PER_PLAYER_POSSIBLE` does not match
                            the length of `NUMBER_OF_PICKED_DIFFERENT_OPERATOR_PER_PLAYER_DISTRIBUTION`.

    :example log output:
    ```
    123e4567-e89b-12d3-a456-426614174000,123e4567-e89b-12d3-a456-426614174111,2,5
    ```

    :corruption: There is a 0.2% chance that some lines in the log will be corrupted, simulating real-world errors.
    """
    logger.info("Generate sample log script start")
    CORRUPTION_PERCENTAGE = 0.2
    NUM_OPERATORS = 71  # Number of unique operators
    MIN_KILLS = 0  # Minimum kills in a game
    MAX_KILLS = 90  # 9 max rounds for ranked with 1 players in each team killing all 5 enemies but going overtime for losing objective
    MIN_UNIQUE_OPERATOR_PER_MATCH_PER_PLAYER = (
        2  # picking the same operator for attacking and defending every round
    )
    MAX_UNIQUE_OPERATORS_PER_MATCH_PER_PLAYER = min(
        9, NUM_OPERATORS
    )  # 9 max rounds for ranked with a different operator on every round

    NUMBER_OF_PICKED_DIFFERENT_OPERATOR_PER_PLAYER_POSSIBLE = [
        x
        for x in range(
            MIN_UNIQUE_OPERATOR_PER_MATCH_PER_PLAYER,
            MAX_UNIQUE_OPERATORS_PER_MATCH_PER_PLAYER + 1,
        )
    ]
    NUMBER_OF_PICKED_DIFFERENT_OPERATOR_PER_PLAYER_DISTRIBUTION = [
        55,
        30,
        10,
        3,
        1.4,
        0.3,
        0.2,
        0.1,
    ]  # Hardcoded approximates
    assert len(NUMBER_OF_PICKED_DIFFERENT_OPERATOR_PER_PLAYER_POSSIBLE) == len(
        NUMBER_OF_PICKED_DIFFERENT_OPERATOR_PER_PLAYER_DISTRIBUTION
    )

    MIN_PLAYERS_IN_A_GAME = 10
    DISTRIBUTION_UNIQUE_PLAYERS_PER_GAME = [
        89.2468,
        4.4920,
        1.9905,
        1.1150,
        0.7097,
        0.4896,
        0.3569,
        0.2707,
        0.2116,
        0.1694,
        0.1381,
        0.1144,
        0.0959,
        0.0812,
        0.0693,
        0.0596,
        0.0516,
        0.0449,
        0.0392,
        0.0343,
        0.0301,
        0.0265,
        0.0233,
        0.0206,
        0.0181,
        0.0159,
        0.0140,
        0.0123,
        0.0107,
        0.0093,
        0.0080,
        0.0069,
        0.0058,
        0.0049,
        0.0040,
        0.0032,
        0.0024,
        0.0018,
        0.0011,
        0.0004,
    ]
    POSSIBILITIES_UNIQUE_PLAYERS_PER_GAME = [
        i
        for i in range(
            MIN_PLAYERS_IN_A_GAME,
            MIN_PLAYERS_IN_A_GAME + len(DISTRIBUTION_UNIQUE_PLAYERS_PER_GAME),
        )
    ]

    operator_ids = [i for i in range(NUM_OPERATORS)]
    player_uuids = [str(uuid.uuid4()) for _ in range(num_players)]
    start_time = time.time()
    for date in dates.split(","):
        output_file = f"r6-matches-{date}.log"
        logger.info(
            f"Generating '{output_file} with {num_matches} matches for {num_players} unique players'"
        )
        with open(output_file, "w") as f:
            for match_num in range(num_matches):
                if match_num % 1000000 == 0 and match_num != 0:
                    logger.info(
                        f"{match_num} matches processed with elapsed time {time.time()-start_time}"
                    )
                player_ids = get_unique_players_per_game_based_on_distribution(
                    player_uuids,
                    POSSIBILITIES_UNIQUE_PLAYERS_PER_GAME,
                    DISTRIBUTION_UNIQUE_PLAYERS_PER_GAME,
                )
                player_operators = assign_operators_to_players(
                    player_ids,
                    operator_ids,
                    NUMBER_OF_PICKED_DIFFERENT_OPERATOR_PER_PLAYER_POSSIBLE,
                    NUMBER_OF_PICKED_DIFFERENT_OPERATOR_PER_PLAYER_DISTRIBUTION,
                )
                kill_distribution = distribute_kills(
                    len(player_operators), random.randint(MIN_KILLS, MAX_KILLS)
                )
                match_id = str(uuid.uuid4())  # Generate a unique match ID
                for (player_id, operator_id), nb_kills in zip(
                    player_operators, kill_distribution
                ):
                    if random.uniform(0, 100) < CORRUPTION_PERCENTAGE:
                        # example of corrupt line with missing field and wrong order fields
                        f.write(f"{nb_kills},%^&*NH&*HJ,{operator_id}\n")
                    else:
                        f.write(f"{player_id},{match_id},{operator_id},{nb_kills}\n")
        logger.info(f"Log file '{output_file}' generated successfully")
    logger.info(
        f"Finished generating all files with elapsed time {time.time()-start_time}"
    )


def assign_operators_to_players(
    player_ids: list[str],
    operator_ids: list[int],
    possibilities_of_unique_players_per_game: list[int],
    distribution_of_unique_players_per_game: list[float],
) -> list[tuple[str, int]]:
    """
    Assigns operators to players in a match based on the distribution of unique players per game.

    This function pairs each player from `player_ids` with a randomly selected operator from `operator_ids`. The number of operators assigned to each player is determined by a probability distribution, given by `distribution_of_unique_players_per_game`, and a set of possible numbers of unique players per game, given by `possibilities_of_unique_players_per_game`. The function returns a list of tuples, where each tuple contains a `player_id` and an assigned `operator_id`.

    :param player_ids: A list of player UUIDs to whom operators will be assigned.
    :type player_ids: list[str]
    :param operator_ids: A list of operator IDs from which to randomly select assignments.
    :type operator_ids: list[int]
    :param possibilities_of_unique_players_per_game: A list of possible numbers of unique operators that can be assigned to players.
    :type possibilities_of_unique_players_per_game: list[int]
    :param distribution_of_unique_players_per_game: The probability distribution for assigning a number of unique operators per game.
    :type distribution_of_unique_players_per_game: list[float]

    :return: A list of tuples, each containing a player UUID and a randomly assigned operator ID.
    :rtype: list[tuple[str, int]]
    """
    return [
        (
            player_id,
            operator,
        )
        for player_id in player_ids
        for operator in random.sample(
            operator_ids,
            random.choices(
                possibilities_of_unique_players_per_game,
                weights=distribution_of_unique_players_per_game,
                k=1,
            )[0],
        )
    ]


def distribute_kills(nb_players: int, nb_of_kills_total: int) -> list[int]:
    """
    Distributes a total number of kills among a given number of players.

    This function uses numpy's multinomial distribution to randomly assign the total number of kills (`nb_of_kills_total`)
    across the specified number of players (`nb_players`). Each player is equally likely to receive any number of kills,
    and the result is a list representing the number of kills for each player.

    :param nb_players: The number of players among whom the kills are to be distributed.
    :type nb_players: int
    :param nb_of_kills_total: The total number of kills to be distributed among the players.
    :type nb_of_kills_total: int

    :return: A list where each element represents the number of kills assigned to each player.
    :rtype: list[int]
    """
    # Use numpy's multinomial to distribute nb_of_kills_total among nb_players
    distribution = np.random.multinomial(
        nb_of_kills_total, [1 / nb_players] * nb_players
    )
    return distribution.tolist()


def get_unique_players_per_game_based_on_distribution(
    player_uuids: list[str],
    possibilities_of_unique_player_per_game: list[int],
    distribution_of_unique_players_per_game: list[float],
) -> list[str]:
    """
    Selects a random subset of unique player UUIDs for a match based on a given distribution of unique players per game.

    The number of unique players in a match can vary between 10 and 40, due to players leaving and others joining
    in casual games. The function ensures that no more than 8 players leave a match, as 5 departures would result
    in an abandoned game. The number of unique players in a match is determined by the provided distribution,
    and the function returns a random sample of player UUIDs from the input list.

    :param player_uuids: A list of player UUIDs from which to sample unique players for the game.
    :type player_uuids: list[str]
    :param possibilities_of_unique_player_per_game: A list of possible numbers of unique players that can be present in a match.
    :type possibilities_of_unique_player_per_game: list[int]
    :param distribution_of_unique_players_per_game: The probability distribution corresponding to the number of unique players in a match.
    :type distribution_of_unique_players_per_game: list[float]

    :return: A list of randomly selected unique player UUIDs for a match, based on the given distribution.
    :rtype: list[str]
    """
    return random.sample(
        player_uuids,
        random.choices(
            possibilities_of_unique_player_per_game,
            weights=distribution_of_unique_players_per_game,
            k=1,
        )[0],
    )


def read_players_file(file_path: str) -> list[str]:
    """
    Reads a file containing player UUIDs and returns them as a list of strings.

    This function opens the specified file, reads each line, strips any surrounding whitespace or newline characters,
    and stores the resulting UUIDs in a list.

    :param file_path: The path to the file containing player UUIDs.
    :type file_path: str

    :return: A list of player UUIDs as strings.
    :rtype: list[str]

    :raises FileNotFoundError: If the file does not exist at the specified `file_path`.
    """
    # Read UUIDs from the file and store them in a list
    logger.info(f"Reading player file '{file_path}'")
    with open(file_path, "r") as f:
        player_uuids = [line.strip() for line in f.readlines()]
    return player_uuids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate R6 Siege sample match logs.")
    parser.add_argument(
        "--players", type=int, default=35000000, help="Number of players"
    )
    parser.add_argument(
        "--matches", type=int, default=30000000, help="Number of matches per day"
    )
    parser.add_argument(
        "--dates",
        type=str,
        default="20240913,20240912,20240911,20240910,20240909,20240908,20240907",
        help="Comma separated dates in format YYYYMMDD,YYYYMMDD,YYYYMMDD",
    )

    args = parser.parse_args()

    generate_log_file(args.players, args.matches, args.dates)
