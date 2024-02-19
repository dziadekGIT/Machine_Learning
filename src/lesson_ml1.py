import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree


def get_example_dataset(big_men: bool = False):
    """
    Prepares a dataset with features 'weight' and 'height', and target 'sex'.
    """

    # Sample sizes for each class (men and women)
    n_samples = 50

    # Normal distribution parameters for men and women
    # Assuming mean weight (in kg) and height (in cm)
    mean_weight_height_men = [75, 175]  # Mean weight and height for men
    if big_men:
        mean_weight_height_men = [95, 195]
    std_weight_height_men = [10, 6]  # Standard deviation for men

    mean_weight_height_women = [65, 160]  # Mean weight and height for women
    std_weight_height_women = [8, 5]  # Standard deviation for women
    np.random.seed(42)
    # Generating samples
    weight_height_men = np.random.normal(
        mean_weight_height_men, std_weight_height_men, (n_samples, 2)
    )
    weight_height_women = np.random.normal(
        mean_weight_height_women, std_weight_height_women, (n_samples, 2)
    )

    # Creating DataFrame
    df_men = pd.DataFrame(weight_height_men, columns=["weight", "height"])
    df_women = pd.DataFrame(weight_height_women, columns=["weight", "height"])

    # Adding 'sex' column
    df_men["sex"] = "men" if not big_men else "men"
    df_women["sex"] = "women"

    # Concatenating the two DataFrames
    df = pd.concat([df_men, df_women], axis=0).reset_index(drop=True)

    return df


def visualize_dataset(df):
    """
    Visualizes the dataset as a scatterplot
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot for men and women
    men = df[df["sex"] == "men"]
    women = df[df["sex"] == "women"]

    plt.scatter(men["weight"], men["height"], color="blue", label="Men")
    plt.scatter(women["weight"], women["height"], color="green", label="Women")

    plt.title("Scatter Plot of Weight vs Height by Sex")
    plt.xlabel("Weight (kg)")
    plt.ylabel("Height (cm)")
    plt.legend()
    plt.show()


def visualize_dataset_with_centroids_and_separator(df):
    """
    Visualizes the dataset with centroids and a separating line using NearestCentroid classifier
    """
    # Prepare data for the classifier
    X = df[["weight", "height"]]
    y = df["sex"].apply(
        lambda x: 0 if x == "men" else 1
    )  # Encode 'men' as 0 and 'women' as 1

    # Initialize and fit the NearestCentroid classifier
    clf = NearestCentroid()
    clf.fit(X, y)

    # Extract centroids
    centroids = clf.centroids_

    # Plot
    plt.figure(figsize=(10, 6))

    # Scatter plot for men and women
    men = df[df["sex"] == "men"]
    women = df[df["sex"] == "women"]

    plt.scatter(men["weight"], men["height"], color="blue", label="Men")
    plt.scatter(women["weight"], women["height"], color="green", label="Women")

    # Plot centroids
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        color="red",
        marker="x",
        s=100,
        label="Centroids",
    )

    # Calculate and plot the separating line
    # The separating line is perpendicular to the line connecting the centroids
    # and passes through the midpoint of the centroids
    mid_point = np.mean(centroids, axis=0)
    slope = (centroids[1, 1] - centroids[0, 1]) / (centroids[1, 0] - centroids[0, 0])
    perp_slope = -1 / slope  # Perpendicular slope
    intercept = mid_point[1] - perp_slope * mid_point[0]

    # Generate x and y values for the line
    x_values = np.linspace(min(df["weight"]), max(df["weight"]), 100)
    y_values = perp_slope * x_values + intercept

    plt.plot(x_values, y_values, color="black", linestyle="--", label="Separating Line")

    plt.title("Scatter Plot with Centroids and Separating Line")
    plt.xlabel("Weight (kg)")
    plt.ylabel("Height (cm)")
    plt.legend()
    plt.show()


def visualize_dataset_sklearn(X, y, marker, name, colours):
    """
    Visualizes the dataset (numpy arrays X and y as used in sklearn) as a scatter plot.

    Parameters:
    X: Feature array.
    y: Label array.
    marker: Marker style for the plot.
    name: Base name for the labels.
    colours: A collection of colours to be used cyclically for different classes.
    """
    # Unique classes in y
    classes = np.unique(y)

    # Iterate through each class to plot
    for i, label in enumerate(classes):
        # Select data belonging to the current class
        class_data = X[y == label]
        # Use cyclic colour for the class
        colour = colours[i % len(colours)]

        # Scatter plot for the class
        plt.scatter(
            class_data[:, 0],
            class_data[:, 1],
            color=colour,
            marker=marker,
            label=f"{name}_{label}",
        )


def exp_data_explanation(X_train, y_train, X_test, y_test, class_known=True):
    """
    data explanation experiment
    """
    visualize_dataset_sklearn(
        X_train, y_train, marker="o", name="train", colours=["blue", "green"]
    )

    if class_known:
        visualize_dataset_sklearn(
            X_test, y_test, marker="x", name="test", colours=["blue", "green"]
        )
    else:
        visualize_dataset_sklearn(
            X_test,
            np.zeros(len(X_test), dtype=np.int32),
            marker="x",
            name="?",
            colours=["red", "orange"],
        )

    plt.grid()
    plt.title(f"Data and classes")
    plt.xlabel("Weight")
    plt.ylabel("Height")
    plt.legend()
    plt.show()
    plt.close()
    plt.show()


if __name__ == "__main__":
    # Call the function to create dataset and visualize it
    df = get_example_dataset(big_men=True)

    # Visualize the example dataset
    if False:
        visualize_dataset(df)

    # Visualize the dataset with centroids and separator
    if False:
        visualize_dataset_with_centroids_and_separator(df)

    # examples
    X = df[["weight", "height"]].values
    # labels
    y = df["sex"].values
    y = LabelEncoder().fit_transform(y)  # here men will be 0, women 1
    # print(y)

    # create some new data
    X_test = np.array([[100, 200], [70, 190], [80, 179], [65, 160], [55, 150]])
    y_test = np.array([0, 0, 0, 1, 1])

    if False:
        exp_data_explanation(X, y, X_test, y_test, False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    if False:
        # print(X_train)
        print(y_train)
        # print(X_test)
        print(y_test)
        exp_data_explanation(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            class_known=False,
        )

    if False:
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        print(y_test)
        print(y_pred)
        print(accuracy_score(y_test, y_pred))
        exp_data_explanation(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_pred,
            class_known=True,
        )

    if False:
        clf = tree.DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(y_test)
        print(y_pred)
        print(accuracy_score(y_test, y_pred))
        exp_data_explanation(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_pred,
            class_known=True,
        )
        plt.close()
        tree.plot_tree(
            clf, feature_names=["weight", "height"], class_names=["men", "women"]
        )
        plt.show()
