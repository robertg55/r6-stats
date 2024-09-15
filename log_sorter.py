import os
import heapq
import argparse
from log_util import setup_logger, delete_directory

logger = setup_logger()


def process_trim_line(line: str) -> str:
    """
    Processes a line from a log file by trimming the first field, if present.

    This function splits the line at the first comma and returns the portion after the comma. If there is no comma,
    it returns the entire line.

    :param line: A string representing a line from the log file.
    :return: The portion of the line after the first comma, or the entire line if no comma is found.
    """
    # Split the line on the first comma and return the part after the comma
    return line.split(",", 1)[1] if "," in line else line


def split_file(
    file_path: str, chunk_size: int, output_dir: str, trim_first_field: bool
) -> list[str]:
    """
    Splits a large file into smaller sorted chunks.

    This function reads a large file and splits it into smaller chunks, each sorted in memory. Optionally, the first field of each line
    can be trimmed. Each chunk is written to a temporary file in the specified output directory.

    :param file_path: The path to the large file to split.
    :param chunk_size: The size of each chunk in bytes.
    :param output_dir: The directory where the chunk files will be saved.
    :param trim_first_field: If True, trims the first field from each line before processing.
    :return: A list of paths to the sorted chunk files.
    """
    chunk_files = []
    with open(file_path, "r") as f:
        lines = []
        for line in f:
            if trim_first_field:
                processed_line = process_trim_line(line)
            else:
                processed_line = line
            lines.append(processed_line)
            if len(lines) * len(processed_line.encode("utf-8")) >= chunk_size:
                # Sort the chunk in memory and write to a temporary file
                lines.sort()
                chunk_file_path = os.path.join(
                    output_dir, f"chunk_{len(chunk_files)}.txt"
                )
                with open(chunk_file_path, "w") as chunk_file:
                    chunk_file.writelines(lines)
                chunk_files.append(chunk_file_path)
                lines = []
        # Write remaining lines if any
        if lines:
            lines.sort()
            chunk_file_path = os.path.join(output_dir, f"chunk_{len(chunk_files)}.txt")
            with open(chunk_file_path, "w") as chunk_file:
                chunk_file.writelines(lines)
            chunk_files.append(chunk_file_path)
    return chunk_files


def merge_files(chunk_files: list[str], output_file: str) -> None:
    """
    Merges multiple sorted chunk files into a single sorted output file.

    This function uses a heap to efficiently merge sorted lines from multiple chunk files and writes the merged result to the
    specified output file.

    :param chunk_files: A list of file paths to the sorted chunk files.
    :param output_file: The path to the final merged output file.
    """
    open_files = [open(chunk_file, "r") for chunk_file in chunk_files]
    with open(output_file, "w") as out:
        # Use heapq to merge sorted lines from each chunk
        sorted_lines = heapq.merge(*open_files)
        out.writelines(sorted_lines)
    # Close all chunk files
    for file in open_files:
        file.close()


def sort_file(
    file_name: str,
    output_file_name: str,
    process_name: str = "default",
    trim_first_field: bool = False,
) -> None:
    """
    Sorts a large file by splitting it into sorted chunks and then merging those chunks.

    This function sorts a large log file by splitting it into smaller chunks, sorting each chunk in memory, and then merging
    the chunks into a single sorted output file. If an already sorted file is found, it skips the sorting. Optionally, the
    first field of each line can be trimmed before sorting.

    :param file_name: The path to the file to be sorted.
    :param output_file_name: The path to the output file where the sorted result will be written.
    :param process_name: A string used to name the temporary chunk files.
    :param trim_first_field: If True, trims the first field from each line before sorting.
    """
    logger.info(f"sorting {file_name}")

    if os.path.isfile(os.path.join(os.getcwd(), output_file_name)):
        logger.info("already sorted file found")
        return

    # Define file paths and sizes
    output_dir = f"chunks-{process_name}"
    os.makedirs(output_dir, exist_ok=True)
    chunk_size = 1024 * 1024 * 512  # 0.5GB

    # Step 1: Split and sort chunks
    logger.info("separating chunk files")
    chunk_files = split_file(file_name, chunk_size, output_dir, trim_first_field)

    # Step 2: Merge sorted chunks into one final sorted file
    logger.info("merging chunk files")
    merge_files(chunk_files, output_file_name)

    # delete the temporary chunk files after sorting
    delete_directory(output_dir)
    logger.info(f"finished sorting {file_name}, writing output to {output_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sort a log file by splitting and merging sorted chunks."
    )
    parser.add_argument(
        "--files-to-consume",
        type=str,
        help="Comma-separated list of file names to process.",
        required=True,
    )
    parser.add_argument(
        "--files-to-produce",
        type=str,
        help="Comma-separated list of file names to produce.",
        required=True,
    )
    parser.add_argument(
        "--process-name",
        type=str,
        help="Name to be used for temporary chunk files.",
        default="default",
    )
    parser.add_argument(
        "--trim-first-field",
        action="store_true",
        help="If set, trims the first field of each line before sorting.",
    )

    args = parser.parse_args()

    # Handle multiple files-to-consume and files-to-produce
    files_to_consume = args.files_to_consume.split(",")
    files_to_produce = args.files_to_produce.split(",")

    # Ensure the number of input and output files match
    if len(files_to_consume) != len(files_to_produce):
        raise ValueError(
            "The number of input files must match the number of output files."
        )

    # Call sort_file for each pair of input and output file
    for input_file, output_file in zip(files_to_consume, files_to_produce):
        sort_file(
            file_name=input_file.strip(),
            output_file_name=output_file.strip(),
            process_name=args.process_name,
            trim_first_field=args.trim_first_field,
        )
