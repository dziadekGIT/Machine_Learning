"""
Task 7 - Titanic  - Prepares titanic data from comma separated values file for plots.
"""
import pandas as pd


PATH = "../datasets/"


def titanic_data(fname="titanic.csv")->pd.DataFrame:
    """
    Function loads Titanic dataset into a DataFrame.

    Parameters:
     - filename (str) : "titanic.csv"

    Returns:
     - DataFrame : df_titanic_data
    """
    df_titanic_data = pd.read_csv(f'{PATH}{fname}')
    #print(df_titanic_data)
    return df_titanic_data

