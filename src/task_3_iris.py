"""
task_3_iris
"""
import pandas as pd


PATH = "../datasets/"


def get_iris(fname="iris.data"):
    """
    function reads file../iris.data and creates DataFrame = df_iris
    function sets column names

    Parameters:
    - fname: default file iris.data

    Returns:
    - df_iris: Returns DataFrame with new column names.   
    """
    df_iris = pd.read_csv(f"{PATH}{fname}", header=None)
    df_iris.columns = [
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
        "species",
    ]

    return df_iris


def analyse_iris() -> None:
    """
    Function analyse irisys from provided DataFrame and prints mean values of particular irisys parts.

    Parameteres: None
    Returns: None
    """
    df_iris_analyze = get_iris()

    print(str(df_iris_analyze["species"].unique()) + "\n")

    df_setosa = df_iris_analyze[df_iris_analyze["species"] == "Iris-setosa"]
    df_versicolor = df_iris_analyze[df_iris_analyze["species"] == "Iris-versicolor"]
    df_virginica = df_iris_analyze[df_iris_analyze["species"] == "Iris-virginica"]

    print("Ilość krokusów Iris Setosa : " + str(len(df_setosa)))
    print("Ilość krokusów Iris Versicolor : " + str(len(df_versicolor)))
    print("Ilość krokusów Iris Virginica : " + str(len(df_virginica)) + "\n")

    print(
        "Średnia długość działki kielicha krokusów Iris Setosa: "
        + str(df_setosa.iloc[:, 0].mean())
    )
    print(
        "Średnia długość działki kielicha krokusów Iris Versicolor: "
        + str(df_versicolor.iloc[:, 0].mean())
    )
    print(
        "Średnia długość działki kielicha krokusów Iris Virginica: "
        + str(df_virginica.iloc[:, 0].mean())
        + "\n"
    )

    print(
        "Średnia szerokość działki kielicha krokusów Iris Setosa: "
        + str(df_setosa.iloc[:, 1].mean())
    )
    print(
        "Średnia szerokość działki kielicha krokusów Iris Versicolor: "
        + str(df_versicolor.iloc[:, 1].mean())
    )
    print(
        "Średnia szerokość działki kielicha krokusów Iris Virginica: "
        + str(df_virginica.iloc[:, 1].mean())
        + "\n"
    )

    print(
        "Średnia długość płatka krokusów Iris Setosa: "
        + str(df_setosa.iloc[:, 2].mean())
    )
    print(
        "Średnia długość płatka kielicha krokusów Iris Versicolor: "
        + str(df_versicolor.iloc[:, 2].mean())
    )
    print(
        "Średnia długość płatka kielicha krokusów Iris Virginica: "
        + str(df_virginica.iloc[:, 2].mean())
        + "\n"
    )

    print(
        "Średnia szerokość płatka krokusów Iris Setosa: "
        + str(df_setosa.iloc[:, 3].mean())
    )
    print(
        "Średnia szerokość płatka kielicha krokusów Iris Versicolor: "
        + str(df_versicolor.iloc[:, 3].mean())
    )
    print(
        "Średnia szerokość płatka kielicha krokusów Iris Virginica: "
        + str(df_virginica.iloc[:, 3].mean())
        + "\n"
    )


if __name__ == "__main__":
  
    analyse_iris()
  