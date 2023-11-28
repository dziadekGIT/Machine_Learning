"""
data loading utilities
"""


def get_file_as_text(filename: str) -> list[str]:
    """
    loads file as a list of sencences

    Parameters:
        filename: name of the file to read
    Returns:
        list of sentences (strings)
    """
    lines = []
    # todo: Pylint error!
    with open(filename) as file:
        for line in file:
            line = line.strip()
            lines.append(line)
    return lines
