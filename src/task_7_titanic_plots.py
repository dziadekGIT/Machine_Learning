"""
Task 7 - Titanic Plots - Plots in matplotlib for titanic data analysis.
"""
import matplotlib.pyplot as plt
from task_7_titanic import titanic_data


def plot_0_passengers_count() -> plt.figure:
    """
    Function plots amount of titanic passangers described by gender, age and class.

    Parameters:
     - None

    Returns:
     - Barplot : fig
    """
    df_titanic = titanic_data()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    genders = df_titanic["Sex"].unique()
    for gender in genders:
        passengers = df_titanic[df_titanic["Sex"] == gender]["PassengerId"].count()
        ax[0].bar(f"{gender.title()} Passengers", passengers)
    ax[0].bar(
        f"Non-Adult Passengers",
        df_titanic[df_titanic["Age"] < 18]["PassengerId"].count(),
    )
    ax[0].set_ylabel("Passengers Count")
    ax[0].set_title("Passengers Count by Gender")

    p_classes = df_titanic["Pclass"].unique()
    for p_class in p_classes:
        passangers = df_titanic[df_titanic["Pclass"] == p_class]["PassengerId"].count()
        ax[1].bar(f"Class {p_class}", passangers)
        ax[1].set_ylabel("Passengers Count")
        ax[1].set_title("Passengers Count by Class")

    plt.tight_layout()
    plt.show()
    plt.close()
    return fig


# def plot_1_passengers_survivability_gender() -> plt.figure:
#     """
#     Function plots survival rate by gender.

#     Parameters:
#      - None

#     Returns:
#      - Barplot : fig
#     """
#     df_titanic = titanic_data()
#     dead_or_alive = df_titanic["Survived"].unique()
#     genders = df_titanic["Sex"].unique()

#     fig, ax = plt.subplots(1, 2, figsize=(15, 5))
#     for person in dead_or_alive:
#         for j, gender in enumerate(genders):
#             passengers = df_titanic[
#                 (df_titanic["Survived"] == person) & (df_titanic["Sex"] == gender)
#             ].count()
#             if person == 0:
#                 status = "Survived"
#             else:
#                 status = "Deceased"
#             ax[j].bar(f"{status}", passengers)
#             ax[j].set_title(f"Gender: {gender.title()}")

#     plt.tight_layout()
#     plt.show()
#     plt.close()
#     return fig


def plot_1_passengers_survivability_gender() -> plt.figure:
    """
    Function plots survival rate by gender.
    Returns:
     - Barplot : fig
    """
    df_titanic = titanic_data()
    dead_alive = df_titanic.groupby(["Sex", "Survived"]).count()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    categories = [("female", 1), ("female", 0), ("male", 1), ("male", 0)]
    for category in categories:
        gender = category[0]
        label = f"{category[0]} ({'Survived' if category[1] == 1 else 'Deceased'})"
        if gender == "female":
            ax[0].bar([label], dead_alive.loc[category].values[0])
        else:
            ax[1].bar([label], dead_alive.loc[category].values[0])

    plt.tight_layout()
    return fig


def plot_2_non_adults_survivability() -> plt.figure:
    """
    Function plots survival rate by adulthood.

    Parameters:
     - None

    Returns:
     - Barplot : fig
    """
    df_titanic = titanic_data()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    non_adults = df_titanic[df_titanic["Age"] < 18]
    adults = df_titanic[df_titanic["Age"] > 18]
    dead_or_alive = df_titanic["Survived"].unique()

    for person in dead_or_alive:
        passengers = non_adults[non_adults["Survived"] == person]["PassengerId"].count()
        status = "Deceased" if person == 0 else "Survived"
        ax[0].bar(status, passengers, label=f"Survived: {person}")
        ax[0].set_title("Non-adult Survival rate")

    for person in dead_or_alive:
        passengers = adults[adults["Survived"] == person]["PassengerId"].count()
        status = "Deceased" if person == 0 else "Survived"
        ax[1].bar(status, passengers, label=f"Survived: {person}")
        ax[1].set_title("Adult Survival rate")

    plt.tight_layout()
    plt.show()
    plt.close()
    return fig


def plot_3_passangers_class_survivability() -> plt.figure:
    """
    Function plots survival rate by boarded class.

    Parameters:
     - None

    Returns:
     - Barplot : fig
    """
    df_titanic = titanic_data()
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    p_class = df_titanic["Pclass"].unique()

    dead_or_alive = df_titanic["Survived"].unique()

    for i in p_class:
        for j in dead_or_alive:
            persons = df_titanic[
                (df_titanic["Pclass"] == i) & (df_titanic["Survived"] == j)
            ].count()
            status = "Deceased" if j == 0 else "Survived"
            ax.bar(f"Class {i} - {status}", persons["PassengerId"])

    plt.tight_layout()
    plt.show()
    plt.close()
    return fig


def plot_4_age_survivalibity_histograms() -> plt.figure:
    """
    Function plots histograms of survival rate by age.

    Parameters:
     - None

    Returns:
     - Histogram plot : fig
    """
    df_titanic = titanic_data()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    dead_or_alive = df_titanic["Survived"].unique()

    for i in dead_or_alive:
        persons = df_titanic[df_titanic["Survived"] == i]["Age"]
        status = "deceased" if i == 0 else "survived"
        ax[i].hist(persons, bins=20)
        ax[i].set_title(f"Titanic {status} people histogram")
        ax[i].set_ylabel("Passengers Count")

    plt.tight_layout()
    plt.show()
    plt.close()
    return fig


def plot_5_fare_survivability_histograms() -> plt.figure:
    """
    Function plots histograms of survival rate by boarded fare.

    Parameters:
     - None

    Returns:
     - Histogram plot : fig
    """
    df_titanic = titanic_data()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    dead_or_alive = df_titanic["Survived"].unique()

    for i in dead_or_alive:
        persons = df_titanic[df_titanic["Survived"] == i]["Fare"]
        status = "deceased" if i == 0 else "survived"
        ax[i].hist(persons, bins=20)
        ax[i].set_title(f"Titanic {status} people histogram")
        ax[i].set_ylabel("Passengers Count")
        ax[i].set_xlabel("Fare")

    plt.tight_layout()
    plt.show()
    plt.close()
    return fig


def plot_6_gender_survival_chance() -> plt.figure:
    """
    Function plots percentage of man and women who survived.

    Parameters:
     - None

    Returns:
     - Barplot : fig
    """
    df_titanic = titanic_data()

    fmale_survived = df_titanic[
        (df_titanic["Sex"] == "female") & (df_titanic["Survived"] == 1)
    ].count()
    male_survived = df_titanic[
        (df_titanic["Sex"] == "male") & (df_titanic["Survived"] == 1)
    ].count()
    prct_fmale_survived = (100 * fmale_survived) / df_titanic[
        (df_titanic["Sex"] == "female")
    ].count()
    prct_male_survived = (100 * male_survived) / df_titanic[
        (df_titanic["Sex"] == "male")
    ].count()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(["Female survivals"], prct_fmale_survived, color="#F59276")
    ax.bar(["Male survivals"], prct_male_survived, color="#84B5A7")
    ax.set_title("Titanic survivalibity percentage")
    ax.set_ylabel("Percentage")

    plt.tight_layout()
    plt.show()
    plt.close()
    return fig


def plot_7_mean_fare_class() -> plt.figure:
    """
    Function plots arithmetic average of fees according to class.

    Parameters:
     - None

    Returns:
     - Barplot : fig
    """
    df_titanic = titanic_data()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    p_class = df_titanic["Pclass"].unique()

    for i in p_class:
        mean_fare = df_titanic[df_titanic["Pclass"] == i]["Fare"].mean()
        c_name = ["First", "Second", "Third"]
        ax.bar(f"Class: {c_name[i-1]}", mean_fare)
    plt.tight_layout()
    plt.show()
    plt.close()
    return fig


def plot_8_fare_and_survival_scatterplots() -> plt.figure:
    """
    Function scatterplots fare and survivors by gender.

    Parameters:
     - None

    Returns:
     - Scatterplot: fig
    """
    df_titanic = titanic_data()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(df_titanic["PassengerId"], df_titanic["Fare"], color="#F59276")
    ax[0].set_xlabel("Passenger ID number")
    ax[0].set_ylabel("Fare amount")
    ax[0].set_title("Fare")

    genders = df_titanic["Sex"].unique()
    for gender in genders:
        survived_id = df_titanic[df_titanic["Sex"] == gender]["PassengerId"]
        survived_age = df_titanic[df_titanic["Sex"] == gender]["Age"]
        ax[1].scatter(survived_id, survived_age, label=f"{gender.title()} Survived")
    ax[1].set_xlabel("Passenger ID number")
    ax[1].set_ylabel("Passenger Age")
    ax[1].set_title("Survived Passengers")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    plt.close()

    return fig


if __name__ == "__main__":
    # plot_0_passengers_count()
    plot_1_passengers_survivability_gender()
    plt.show()
    plt.close()
    # plot_2_non_adults_survivability()
    # plot_3_passangers_class_survivability()
    # plot_4_age_survivalibity_histograms()
    # plot_5_fare_survivability_histograms()
    # plot_6_gender_survival_chance()
    # plot_7_mean_fare_class()
    # plot_8_fare_and_survival_scatterplots()
