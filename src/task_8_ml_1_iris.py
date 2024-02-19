"""
Task 8 - Machine Learning warmup with sklearn.
"""
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import requests
from bs4 import BeautifulSoup


def iris_dataset() -> pd.DataFrame:
    """
    Prepares a dataset witch features iris.

    Parameters:
    - None

    Returns:
    - pd.DataFrame : Returns dataframe created with iris data form sklearn dataset.
    """
    iris_data = load_iris()

    # pylint: disable=no-member
    df_iris = pd.DataFrame(iris_data.data)
    df_iris.columns = [
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
    ]
    # pylint: disable=no-member
    df_iris.insert(0, "class", iris_data.target)

    # print(df_iris)
    return df_iris


def iris_dataset_fromfile() -> pd.DataFrame:
    """
    Function prepares a dataset with features iris from file.

    Parameters:
    - None

    Returns:
    - pd.DataFrame : Returns dataframe with iris data form file new_iris.npz
    """

    path = "..\\datasets\\new_iris.npz"
    blob = np.load(path)

    iris_data = []
    for f in blob.files:
        iris_data.append(blob[f])
    df_iris = pd.DataFrame(iris_data[0])
    df_iris.columns = [
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
    ]
    df_iris.insert(0, "class", iris_data[1].astype(int))
    # print(df_iris)
    
    return df_iris

def websrapped_data()-> pd.DataFrame:
    """
    Function prepares a dataset from table in https://www.wikidoc.org/index.php/Iris_flower_data_set.

    Parameters:
    - None

    Returns:
    - pd.DataFrame : Returns dataframe with iris data form www.
    """
    page = requests.get('https://www.wikidoc.org/index.php/Iris_flower_data_set').text
    soup = BeautifulSoup(page,"html.parser")
 
    
    df_iris_webscrapped = pd.DataFrame(columns=["class", "sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"])
    rows = soup.find_all('tbody')[0].find_all('tr')
    
    for row in rows:
        columns = row.find_all('td')
        if len(columns) == 5:  
            s_len = columns[0].text 
            s_width = columns[1].text 
            p_len = columns[2].text 
            p_width = columns[3].text 
            cl = columns[4].text.strip() 
            if cl == 'setosa':
                cl_int = 0
            elif cl == 'versicolor':
                cl_int = 1
            elif cl == 'virginica':
                cl_int =2
            df_iris_webscrapped = df_iris_webscrapped._append({
                "sepal_length_cm": s_len,
                "sepal_width_cm": s_width,
                "petal_length_cm": p_len,
                "petal_width_cm": p_width,
                "class": cl_int
            }, ignore_index=True)
    df_iris_webscrapped['class'] = df_iris_webscrapped['class'].astype(int)
    
    return df_iris_webscrapped


def visualize_dataset(df) -> plt.figure:
    """
    Visualizes the dataset as a scatterplot.

    Parameters:
    - pd.Dataframe - dataframe must include columns :
        "sepal_length_cm"
        "sepal_width_cm"
        "petal_length_cm"
        "petal_width_cm"
        "class"

    Returns:
    plt.figure - function returns iris scatterplot figure
    """

    fig, ax = plt.subplots(1, 2, figsize=(25, 5))
    setosa = df[df["class"] == 0]
    versicolor = df[df["class"] == 1]
    virginica = df[df["class"] == 2]

    ax[0].scatter(
        setosa["sepal_length_cm"],
        setosa["sepal_width_cm"],
        color="blue",
        label="Iris Setosa",
    )
    ax[0].scatter(
        versicolor["sepal_length_cm"],
        versicolor["sepal_width_cm"],
        color="red",
        label="Iris Versicolor",
    )
    ax[0].scatter(
        virginica["sepal_length_cm"],
        virginica["sepal_width_cm"],
        color="green",
        label="Iris Virignica",
    )
    ax[0].set_title("Scatter Plot of sepals lenght/width")
    ax[0].set_xlabel("Sepal length (cm)")
    ax[0].set_ylabel("Sepal width (cm)")
    ax[0].legend()

    ax[1].scatter(
        setosa["petal_length_cm"],
        setosa["petal_width_cm"],
        color="blue",
        label="Iris Setosa",
    )
    ax[1].scatter(
        versicolor["petal_length_cm"],
        versicolor["petal_width_cm"],
        color="red",
        label="Iris Versicolor",
    )
    ax[1].scatter(
        virginica["petal_length_cm"],
        virginica["petal_width_cm"],
        color="green",
        label="Iris virignica",
    )
    ax[1].set_title("Scatter Plot of petals lenght/width")
    ax[1].set_xlabel("Petal length (cm)")
    ax[1].set_ylabel("Petal width (cm)")
    ax[1].legend()

    return fig


def data_preparation(df, trait) -> pd.DataFrame:
    """
    Function splits data into training and testing sets form provided datagrame.
    Training data coprises 80% of the dataframe content.

    Parameteres:
    - pd.Dataframe - dataframe must include columns:
        'petal_length_cm'
        'petal_width_cm'
        'class'
    - bool - Boolean value determines if data will be shuffled before split.

    Returns:
    - DataFrame - X_train, y_train, X_test, y_test - df's of train and test values.
    """
    if trait == "petal":
        X = df.iloc[:, 3:]
        y = df.iloc[:, 0]
    elif trait =="sepal" :
        X = df.iloc[:, 1:3]
        y = df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X ,y , train_size=0.8, shuffle=True)
    return X_train, y_train, X_test, y_test

def KNN_classificator(X_train, y_train, X_test, y_test, n_neighbors) -> np.float64:
    """
    Function runs KNeigborsClassifier on provided data.
    
    Parameters:
    - X_train - Train data [(x1,y1), (x2,y2)..... , (xn, yn)]
    - y_train - Train labels data [i1, i2, i3, ...,in]
    - X_test - Test data [(x1,y1), (x2,y2)..... , (xn, yn)]
    - y_test - Train labels data [i1, i2, i3, ...,in]

    Returns:
    - numpy.float64 - returns accuracy of predictions
    """

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm="auto")
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    # print(accuracy)

    return accuracy


def ten_times_test(trait) -> np.float64:
    """
    Runs 10x function KNN_classificator and counts mean value of KN classyfier accuracies.
    
    Parameters:
    - None
    
    Returns:
    - numpy.float64 - mean effectivenes of 10x accuracy tests.
    """
    dataframe = iris_dataset()
    effectiveness = []
    for i in range(1, 11):
        X_train, y_train, X_test, y_test = data_preparation(dataframe, trait)
        effectiveness.append(
            KNN_classificator(X_train, y_train, X_test, y_test, n_neighbors=1)
        )
    # print(effectiveness)
    # print(f'Mean effectiveness of ten time KNN classification is {np.mean(effectiveness)}')
    mean_effectivenes = np.mean(effectiveness)
    return mean_effectivenes


def one_to_ten_KNN_test(trait) -> plt.figure:
    """
    Runs ten times KNN_classificator function increasing neighbors count each iteration.

    Parameters:
    - None
    
    Returns:
    - plt.figure - returns barplot figure with best fit for neighbour count.
    """
    dataframe = iris_dataset()
    effectiveness = []
    for i in range(1, 11):
        X_train, y_train, X_test, y_test = data_preparation(dataframe, trait)
        effectiveness.append(
            KNN_classificator(X_train, y_train, X_test, y_test, n_neighbors=i)
        )

    # print(effectiveness)
    best_index = effectiveness.index(max(effectiveness))
    effectiveness_list = list(effectiveness)
    # print(best_index)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    x_labels = [f"Nbr: {i}" for i in range(1, 11)]
    ax.bar(x_labels, effectiveness_list)
    ax.set_ylabel("Percentage of effciency")
    ax.set_xlabel("Number of neighbours for KNN classifier")
    ax.patches[best_index].set_facecolor("red")
    ax.patches[best_index].set_label("best fit")
    ax.legend()
    return fig


def KNN_scatterplot(dataframe) -> plt.figure:
    """
    Function creates scatterplots for KN classifier with train and test data,
    determines correct and incorrect predictions.

    Paremeters:
    - pd.Dataframe - dataframe must include columns :
        "petal_length_cm"
        "petal_width_cm"
        "class"

    Returns:
    -  plt.figue - Returns scatterplot figure.
    """

    X_train, y_train, X_test, y_test = data_preparation(dataframe, trait='petal')

    
    setosa = X_train[y_train==0]
    versicolor = X_train[y_train==1]
    virignica = X_train[y_train==2]

    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)

    knn = KNeighborsClassifier(n_neighbors=1, algorithm="auto")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    correct_indices = y_test[y_test == y_pred]
    incorrect_indices = y_test[y_test != y_pred] 

    correct_df = X_test.loc[correct_indices.index]
    incorrect_df = X_test.loc[incorrect_indices.index]
    
    fig, ax = plt.subplots(figsize=(20, 13))
    ax.scatter(setosa['petal_length_cm'], setosa['petal_width_cm'], color="blue", label="Iris Setosa")
    ax.scatter(versicolor['petal_length_cm'], versicolor['petal_width_cm'], color="orange", label="Iris Versicolor")
    ax.scatter(virignica['petal_length_cm'], virignica['petal_width_cm'], color="magenta", label="Iris Virginica")
    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Petal width (cm)")
    ax.scatter(correct_df['petal_length_cm'], correct_df['petal_width_cm'], color="green", marker="x", label="Correct prediction")
    ax.scatter(incorrect_df['petal_length_cm'], incorrect_df['petal_width_cm'], color="red", marker="x", label="Incorrect prediction")

    ax.legend()
    return fig


def KNN_classificator_train_sklearn_test_file() -> np.float64:
    """
    Function train KNeigborsClassifier with sklearn data and tests predictions
    with data provided from file.

    Parameters:
    - None

    Returns:
    - np.float64 - Returns prediction accuracy.
    """

  
    X_train = iris_dataset()[["petal_length_cm","petal_width_cm"]]
    y_train = iris_dataset()["class"]

    X_test = iris_dataset_fromfile()[["petal_length_cm","petal_width_cm"]]
    y_test = iris_dataset_fromfile()["class"]
    # X_test = websrapped_data()[["petal_length_cm","petal_width_cm"]]
    # y_test = websrapped_data()["class"]
  

    knn = KNeighborsClassifier(n_neighbors=1, algorithm="auto")
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy


if __name__ == "__main__":
    # choose dataframe from sklrean.iris or from file, or websrapped data for requested functions
    if False:
        dataframe = iris_dataset()

    if False:
        dataframe = iris_dataset_fromfile()
       
    if True:
        dataframe = websrapped_data()
       


    # requested functions
    if False:
        visualize_dataset(dataframe)
        plt.show()
  
    if False:
        X_train, X_test, y_train, y_test = data_preparation(dataframe, trait='petal')
        KNN_classificator(X_train, y_train, X_test, y_test, n_neighbors=1)
        # print(KNN_classificator(X_train, y_train, X_test, y_test, n_neighbors=1))

    if False:
        mean_accuracy = ten_times_test(trait='petal')
        print(mean_accuracy)

    if False:
        one_to_ten_KNN_test(trait='petal')
        plt.show()

    if False:
        KNN_scatterplot(dataframe)
        plt.show()
    if True:
        accuracy = KNN_classificator_train_sklearn_test_file()
        print(accuracy)
    
  
