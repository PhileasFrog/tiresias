import pathlib
from typing import List, Union

def get_path_from_input(input_path: str, allowed_extensions: List[str]) -> List[pathlib.Path]:
    """
    Get a list of pathlib.Path objects that match the given input path and allowed extensions.

    Args:
        input_path (str): The input file or directory path.
        allowed_extensions (List[str]): A list of allowed file extensions for example `.jpg`.

    Returns:
        List[pathlib.Path]: A list of pathlib.Path objects that match the given criteria.

    Raises:
        FileNotFoundError: If the input file or directory path is not found.
        ValueError: If the input is neither a file path nor a directory path, or if no file with
            the required extension is found in the directory.
    """
    path = pathlib.Path(input_path)
    if not path.exists():
        raise FileNotFoundError('File or Dir path is not found')

    if path.is_file() and path.suffix not in allowed_extensions:
        raise ValueError(f'File extension should be in {allowed_extensions}')

    if path.is_file() and path.suffix in allowed_extensions:
        return [path]

    if path.is_dir():
        paths = [file for file in path.glob("*") if file.suffix in allowed_extensions]
        if not paths:
            raise ValueError(f'No file with required extension {allowed_extensions} in dir {input_path}')
        return paths

    raise ValueError('Input should be a file path or a directory path')

