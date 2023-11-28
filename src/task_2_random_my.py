"""
task_2_random
"""
import re
import names

PATH = "../datasets/"


def find_person_occurrences(input_string: str) -> list:
    """
    Find all occurrences of strings like 'person_1', 'person_2', etc., in the given string.

    Parameters:
    - input_string (str): The string to search in.

    Returns:
    - list: A list of all found occurrences.
    """
    pattern = r"{person_\d+}"
    return re.findall(pattern, input_string)


def read_file(filename: str) -> list[str]:
    """
    smartly reads file
    """
    try:
        with open(filename, encoding="utf-8") as file:
            lines = [line.strip() for line in file]
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    return lines


def save_file(filename: str, lines: list[str]) -> None:
    """
    smartly save file
    """
    name_part, extension = filename.rsplit(".", 1)
    new_filename = f"{name_part}_replaced.{extension}"

    with open(new_filename, "w", encoding="utf") as file:
        file.writelines([l + "\n" for l in lines])


def replace_persons(
    filename: str = "../datasets/problematic_sencences.txt", save: bool = False
) -> list[str]:
    """

    Funcion replacing hooks ( '{person_[0-20]}' ) in txt  to real names.

    Parameters:
     - filename (str) : Source file. Default :  ../datasets/problematic_sencences.txt.
     - save (bool) : Save to file "problematic_sentences_replaced" interface while True.

     Returns:
     - List [str] : Returns List that contains cleaned text.
    """

    lines = read_file(filename)

    replacemens = {}
    for i, l in enumerate(lines):
        pfound = find_person_occurrences(l)
        for pstring in pfound:
            if pstring not in replacemens:
                replacemens[pstring] = names.get_first_name()
            lines[i] = lines[i].replace(pstring, replacemens[pstring])

    if save:
        save_file(filename, lines)

    return lines


if __name__ == "__main__":
    glines = replace_persons(save=True)
    for l in glines:
        print(l)
