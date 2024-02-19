import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import streamlit as st
from sklearn.datasets import load_iris


# Load Iris dataset into a DataFrame
def load_iris_data():
    """
    load ds
    """
    iris = load_iris()
    iris_df = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )
    iris_df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return iris_df


# Scatterplot function
def scatterplot_iris(df):
    """
    scatterplot
    """
    fig, ax = plt.subplots()
    for species in df["species"].unique():
        subset = df[df["species"] == species]
        ax.scatter(
            subset["sepal length (cm)"], subset["sepal width (cm)"], label=species
        )
    ax.legend()
    ax.set_xlabel("Sepal Length (cm)")
    ax.set_ylabel("Sepal Width (cm)")
    ax.set_title("Iris Sepal Length vs Width Scatter Plot")
    return fig


# Barplot function
def barplot_iris(df):
    """
    barplot
    """
    mean_df = df.groupby("species").mean()
    fig, ax = plt.subplots()
    mean_df.plot(kind="bar", ax=ax)
    ax.set_title("Mean Measurements of Iris Species")
    ax.set_ylabel("Measurements (cm)")
    return fig


# Histogram function
def histogram_iris(df):
    """
    histogram
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    columns = df.columns[:4]
    for i, col in enumerate(columns):
        ax = axs[i // 2, i % 2]
        df[col].plot(kind="hist", bins=20, ax=ax)
        ax.set_title(f"Histogram of {col}")
    plt.tight_layout()
    return fig


# Sinus function plot
def sinus_plot(n=100):
    """
    sinus
    """
    x = np.linspace(0, n, 400)
    y = np.sin(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Sinus Function")
    ax.set_xlabel("X")
    ax.set_ylabel("sin(X)")
    return fig


def run():
    """
    plot in matplotlib
    """
    sinus_plot()
    plt.show()
    plt.close()

    df = load_iris_data()
    scatterplot_iris(df)
    plt.show()
    plt.close()

    barplot_iris(df)
    plt.show()
    plt.close()

    histogram_iris(df)
    plt.show()
    plt.close()


if __name__ == "__main__":
    run()
