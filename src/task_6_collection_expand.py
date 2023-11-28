"""
Task 6 - collection expand
"""
import random
import re
import pickle
import pandas as pd
import names


from task_5_monsters_df_operations import create_monsters_df

def collection_expand()->pd.DataFrame:

    """
    Imports dataframes from task_5_monsters_df_operations and expands'em.

    Parametrers : None
    Return : DataFrames
    """
    
    df_dragons_exp, df_minions_exp, df_items_exp = create_monsters_df()
    
    # Adding random dragons to df_dragons as df_dragons_exp
    new_dragon_names = []
    for new_d_name in range(3):
        new_d_name = names.get_first_name()
        while new_d_name in new_dragon_names:
            new_d_name = names.get_first_name()     
        new_dragon_names.append(new_d_name) 
    
    breath = ['fire','acid','electricity','poison','ice']
    colour = ['red','green','blue']
    for new_dragon in new_dragon_names:
        df_dragons_exp.loc[new_dragon] = [random.randint(1,3),random.choice(breath),random.choice(colour)]
    
    
    # Adding some minions to df_minions as df_minions_exp
    for new_minion in range(3):
        new_minion = names.get_first_name()
        new_minion_level = random.randint(1,10)
        df_minions_exp.loc[len(df_minions_exp)] = [random.choice(df_minions_exp['species']), f'{new_minion} (lvl{new_minion_level})',new_minion_level]
    
    # Adding some items to df_items as df_items_exp
    new_magical_items = ['Keyboard','Mouse','Processor','GPU', 'Laptop', 'Webcam']
    new_property = ['error','debugging','multithredding','version control', 'git branching', 'exception crash']
    new_magical_item_names = []
    for new_item in range(3):
        new_item = f'{random.choice(new_magical_items)} of {random.choice(new_property)}'
        while new_item in new_magical_item_names:
            new_item = f'{random.choice(new_magical_items)} of {random.choice(new_property)}'
        new_magical_item_names.append(new_item)
        
    for new_magic_item in new_magical_item_names:
        df_items_exp.loc[new_magic_item] = [random.randint(1,5),random.randrange(50,1000,25)] 
    
    print(f'{df_dragons_exp}\n')
    print(f'{df_minions_exp}\n')
    print(df_items_exp)
    return df_dragons_exp, df_minions_exp, df_items_exp

if __name__ == "__main__":
    collection_expand()