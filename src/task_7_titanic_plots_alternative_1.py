"""
Task 7 - Titanic Plots - Plots in matplotlib for titanic data analysis.
"""
import matplotlib.pyplot as plt
from task_7_titanic import titanic_data


def plot_1_passengers_survivability_gender_new() -> plt.figure:
    """
    Function plots survival rate by gender.
    Returns:
     - Barplot : fig
    """
    df_titanic = titanic_data()
    # new column for sure counting (no NaNs amd good name)
    df_titanic["count"] = 1
    survivability_stats = df_titanic.groupby(["Sex", "Survived"]).count()["count"]
    # trick to turn index into columns
    survivability_stats = survivability_stats.reset_index()
    # trick to translate things into strings
    trans_dict = {1: "Survived", 0: "Deceased"}
    # prepare labels
    survivability_stats["label"] = (
        survivability_stats["Sex"]
        + " "
        + [trans_dict[t] for t in survivability_stats["Survived"]]
    )

    colors = ["Olive", "Green", "Red", "Orange"]
    # see the table
    print(survivability_stats)
    plt.figure(figsize=(8, 6))
    bar_width = 0.5
    for i, row in survivability_stats.iterrows():
        plt.bar(
            i * bar_width, row["count"], bar_width, color=colors[i], label=row["label"]
        )
    plt.legend()
    plt.xticks([])
    # Set the labels and title
    plt.xlabel("Group")
    plt.ylabel("Count")
    plt.title("Survivability on the Titanic by Sex and Survival Status")


if __name__ == "__main__":
    plot_1_passengers_survivability_gender_new()
    plt.show()
    plt.close()
