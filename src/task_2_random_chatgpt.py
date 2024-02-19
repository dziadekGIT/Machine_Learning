"""
task_2_random
"""
import pandas as pd


PATH = "../datasets/"


def get_person(fname="ludzie.data"):
    """

    Importing CSV file to Dataframe function - imports csv file and returns dataframe.

    Parameters:
     - fname : CSV source file. Default :  f"{PATH}{"ludzie.DATA"}.

     Returns:
     - df_people : DataFrame containing people first and second names, and nicknames.
    """

    df_people = pd.read_csv(f"{PATH}{fname}", header=None)
    return df_people


def get_person_name(index: int) -> str:
    """
    Retrieve a person's name based on the index.

    Parameters:
    - index (int): The index of the person's name to retrieve.

    Returns:
    - str: The retrieved person's name.
    """
    return get_person().iloc[index, 0]


def replace_hooks_in_line(line: str, people_dict: dict) -> str:
    """
    Replace hooks in a given line with corresponding person names.

    Parameters:
    - line (str): The line in which to replace hooks.
    - people_dict (dict): A dictionary mapping hooks to person names.

    Returns:
    - str: The line with hooks replaced by person names.
    """
    for key, value in people_dict.items():
        line = line.replace("{" + key + "}", value)
    return line


def read_file_lines(filename: str) -> list:
    """
    Read lines from a file.

    Parameters:
    - filename (str): The name of the file to read from.

    Returns:
    - list: A list of lines read from the file.
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return [line.strip() for line in file]
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return []


def save_to_file(data: str, filename: str):
    """
    Save data to a file.

    Parameters:
    - data (str): Data to be saved.
    - filename (str): The name of the file to save to.
    """
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(data)
    except IOError:
        print(f"Error: Could not write to {filename}.")


def replace_persons(
    filename: str = "../datasets/problematic_sencences.txt", save: bool = False
) -> list:
    """
    Function replacing hooks ( '{person_[0-20]}' ) in a text file with real names.

    Parameters:
    - filename (str): Source file. Default:  ../datasets/problematic_sencences.txt.
    - save (bool): Save to file "problematic_sentences_replaced" if True.

    Returns:
    - List[str]: List that contains cleaned text.
    """
    lines = read_file_lines(filename)
    person_prefix = "person_"
    people_dict = {person_prefix + str(i): get_person_name(i) for i in range(21)}

    replaced_lines = [replace_hooks_in_line(line, people_dict) for line in lines]

    if save:
        save_to_file("\n".join(replaced_lines), "problematic_sentences_replaced.txt")

    return replaced_lines


if __name__ == "__main__":
    # my_dict = {"ala": "ala value"}
    # my_dict = {}
    # my_dict["ala"] = "ala value"

    lines = replace_persons(save=False)
    for l in lines:
        print(lines)
