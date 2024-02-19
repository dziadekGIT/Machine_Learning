"""
Task 1 - Cleaning  "strange poem" and save to file while save = true
"""
import re
from util_data import get_file_as_text


PATH = "../datasets/"


def clean_brackets(
    filename: str = "../datasets/txt_4_regex.txt", save: bool = False, verbose=False
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

    # # simple style
    # cleaned_lines = []
    # for cur_line in f_txt:
    #     cur_line_cleaned = re.sub(r"\[[^\]]*\]|\([^\)]*\)", "", cur_line)
    #     cleaned_lines.append(cur_line_cleaned)

    cleaned_lines = [re.sub(r"\[[^\]]*\]|\([^\)]*\)", "", l) for l in f_txt]

    # #dynamic list creation example
    # collection = [1, 2, 3, 4]
    # created_list = [l + 5 for l in collection]
    # print(created_list)

    if verbose:
        print(cleaned_lines)

    if save:
        with open("my_file_cleaned", "w", encoding="utf") as file:
            file.writelines(cleaned_lines)
            file.close()
    return cleaned_lines


if __name__ == "__main__":
    ret_lines = clean_brackets(save=False)

    if ret_lines is None:
        print("something is wrong")
    else:
        print(ret_lines)
