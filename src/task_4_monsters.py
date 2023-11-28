"""
Task 4 - Monsters!
"""

import pickle
import re
import pprint


def monsters():
    """

    Deserialisation and operations on Monsters dict.
    Function deserialise monsters data dict and prints one headed dragon names,
    species and level more than 3 minions, items with cost more than 100 and power more
    or equal 4.

    Parameters:
     None

     Returns:
     None

    """
    with open("../datasets/monsters.pkl", "rb") as monsters:
        monsters_data = pickle.load(monsters)

    # pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(monsters_data['stuff'])

    for color, attributes in monsters_data["dragons"].items():
        if attributes.get("heads", 0) == 1:
            print(f"Name of one head dragon: {monsters_data['dragons'][color]['name']}")

    pattern_level = re.compile(r"lvl(\d+)")
    for race, names in monsters_data["minions"].items():
        for name in names:
            match = pattern_level.search(name)
            if match:
                level = int(match.group(1))
                if level > 3:
                    print(f"Imię i poziom: {name} Rasa: {race}")

    pattern_magical_power = re.compile(r"'magical_power': (\d+)")
    pattern_name = re.compile(r"'name': '(.*?)'")
    pattern_price = re.compile(r"'price': (\d+)")

    for items in monsters_data["stuff"]:
        if "magical_power" in items:
            items_string = str(items)
            match_mp = pattern_magical_power.search(items_string)
            match_n = pattern_name.search(items_string)
            match_prc = pattern_price.search(items_string)
            if (
                match_mp
                and int(match_mp.group(1)) >= 4
                and int(match_prc.group(1)) > 100
            ):
                print(
                    f"{match_n.group(1)}, MP:{match_mp.group(1)}, price:{match_prc.group(1)}"
                )

    return None


if __name__ == "__main__":
    monsters()
