"""
Task 1 - Cleaning  "strange poem" and save to file while save = true
"""
import re
from util_data import get_file_as_text


PATH = "../datasets/"


def clean_brackets(
    filename: str = "../datasets/txt_4_regex.txt", save: bool = False
) -> list[str]:
    """

    Cleaning function - cleans text of square and round brackets with numbers in between.

    Parameters:
     - filename (str) : Source file path. Default :  ../datasets/txt_4_regex.txt.
     - save (bool) : Save to file "my_file_cleaned" interface while True.

     Returns:
     - List [str] : Returns List that contains cleaned text.


    """
    f_txt = get_file_as_text(filename)

    f_txt_clean = re.sub(r"\[[^\]]*\]|\([^\)]*\)", "", str(f_txt))
    print(f_txt_clean)

    if save:
        with open("my_file_cleaned", "w", encoding="utf") as file:
            file.write(f_txt_clean)
            file.close()


if __name__ == "__main__":
    clean_brackets(save=False)
