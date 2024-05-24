import os
import shutil


def clean_plot_directory(directory_path: str = "plot/") -> None:
    """
    Clean a directory and remove all its content.
    :param directory_path: The path of the directory to be cleaned.
    :return: nothing
    """
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file
                print(f"Removed file: {file_path}")
            elif os.path.isdir(file_path):
                # Rimuove la directory se necessario
                os.rmdir(file_path)
                print(f"Removed directory: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def clean_log_file(file_path: str = "log/output.log") -> None:
    """
    Clean log file.
    :param file_path: The path containing log file to clean.
    :return: nothing
    """

    """
    Remove a file.
    :param file_path: The path of the file to be removed.
    :return: nothing
    """
    if os.path.isfile(file_path):
        os.remove(file_path)
    else:
        raise ValueError(f"The provided path '{file_path}' is not a valid file.")
