"""
exercises, lesson 1
"""
import re
import pandas as pd
from util_data import get_file_as_text

PATH = "../datasets/"


def task1():
    """
    Task 1
    prints the strange poem
    """
    f_txt = get_file_as_text("../datasets/txt_4_regex.txt")
    print(f_txt)
    for line in f_txt:
        print(line)


def task2():
    """
    Task 2
    https://regex101.com/
    """
    pattern = "the ([a-z]+)"
    text = "the first, the second, the third and the last."
    found = re.findall(pattern, text)
    print(found)


def task3(fname="iris.data"):
    """
    Task 3
    """

    # print(PATH + fname)
    df_iris = pd.read_csv(f"{PATH}{fname}")
    # print(df_iris)
    # print(df_iris.iloc[0])
    # print(df_iris.iloc[:, 0])
    # print(df_iris.iloc[:, 0].sum())
    # print(df_iris.iloc[:, :4].mean())
    df_iris.columns = ["f1", "f2", "f3", "f4", "class"]
    print(df_iris)
    print(df_iris.iloc[:, :4].mean(axis=0))
    # df_iris["f1"] = df_iris["f1"] + df_iris["f2"]
    # print(df_iris)


def chat_gpt_solution():
    text = "the first, the second, the third and the last."
    pattern = r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|teenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|last)\b"

    matches = re.findall(pattern, text)

    print(matches)


if __name__ == "__main__":
    # task1()
    # task2()
    # task3()
    chat_gpt_solution()
