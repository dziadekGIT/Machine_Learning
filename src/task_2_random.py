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

def replace_persons(filename:str="../datasets/problematic_sencences.txt",save:bool=False)->list[str]:
    """
     
    Funcion replacing hooks ( '{person_[0-20]}' ) in txt  to real names.

    Parameters:
     - filename (str) : Source file. Default :  ../datasets/problematic_sencences.txt.
     - save (bool) : Save to file "problematic_sentences_replaced" interface while True.

     Returns:
     - List [str] : Returns List that contains cleaned text.
    """

    lines = []
    with open(filename, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            lines.append(line)

    persons =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] 
    person_prefix = "person_"
    people_dict = {}
    
    for i in persons:

        people_dict[person_prefix+str(i)] = get_person().iloc[i-1, 0]
      
    lines_merged = "".join(lines)

    for key, values in people_dict.items():
        lines_merged = lines_merged.replace("{" + key + "}", values)
    
    replaced_hooks = lines_merged.split(".")
    print(replaced_hooks)


    if save:    
        with open("problematic_sentences_replaced", 'w', encoding="utf") as file:
            file.write(str(replaced_hooks))
            file.close()


    return replaced_hooks

if __name__ == "__main__":
    replace_persons(save=False)
   