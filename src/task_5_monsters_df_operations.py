"""
Task 5 - monsters_df_operations
"""
import re
import pickle
import pandas as pd


def get_file_and_deserial(filename: pickle) -> dict:
    """
    Imports and deserilize pickle file. Provide only from trusted resources!
    """
    with open(filename, "rb") as file:
        deserialised_data = pickle.load(file)

    return deserialised_data


def create_monsters_df() -> pd.DataFrame:
    """
    Creates Dataframes for particular keys in provided dictionary,
    organizes them, incresing the readability of individual tables.

    Parametrers : None
    Return : df_dragons,df_minions, df_items
    """

    monsters_data = get_file_and_deserial("../datasets/monsters.pkl")

    # Checks type of data structure
    # for key in monsters_data:
    #     print(f'Typ słownika : {key}  {type(monsters_data[key])}')

    df_dragons = pd.DataFrame(monsters_data["dragons"]).T
    df_dragons["colour"] = df_dragons.index
    df_dragons = df_dragons.set_index("name")

    df_minions = pd.DataFrame(
        [(k, v) for k, values in monsters_data["minions"].items() for v in values],
        columns=["species", "name"],
    )
    pattern_level = re.compile(r"lvl(\d+)")
    level_list = []
    for levels in df_minions.iloc[:, 1]:
        match = pattern_level.search(levels)
        if match:
            level_list.append(match.group(1))
        else:
            level_list.append(None)

    df_minions["level"] = level_list

    df_items = pd.DataFrame(monsters_data["stuff"])
    df_items = df_items.set_index("name")

    return df_dragons, df_minions, df_items


if __name__ == "__main__":
    create_monsters_df()
