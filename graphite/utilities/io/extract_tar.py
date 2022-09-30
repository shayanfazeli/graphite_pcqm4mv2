import tarfile
import os


def extract_tar_gz_file(
        filepath: str,
        output_directory: str = None
):
    if output_directory is None:
        output_directory = os.path.dirname(filepath)

    # open file
    file = tarfile.open(filepath)

    # extracting file
    file.extractall(output_directory)

    # - closing and unlinking the file
    file.close()
    os.unlink(filepath)
