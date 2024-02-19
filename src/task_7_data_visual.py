"""
Task 7 - Titanic data Visualisation - Visualises titanic data with streamlit.
"""
import streamlit as st
from task_7_titanic import titanic_data
from task_7_titanic_plots import (
    plot_0_passengers_count,
    plot_1_passengers_survivability_gender,
    plot_2_non_adults_survivability,
    plot_3_passangers_class_survivability,
    plot_4_age_survivalibity_histograms,
    plot_5_fare_survivability_histograms,
    plot_6_gender_survival_chance,
    plot_7_mean_fare_class,
    plot_8_fare_and_survival_scatterplots,
)


def header(selected_option):
    """
    Function describes title and data plots in header.

    Parameters:
     - selected_option[]  : Figure titles.

    Returns:
     - None
    """

    st.title("Titanic Accident Data Analysis")
    st.subheader(f"Detailed Charts of {selected_option[1]} : {selected_option[0]}")


def main_page(selected_option):
    """
    Function displays selected from sidebar selection figure.

    Parameters:
     - selected_option[]  : Figure titles.

    Returns:
     - None
    """
    if selected_option[0] == "Gender and Class":
        st.pyplot(plot_0_passengers_count())
    elif selected_option[0] == "Gender bar graph":
        st.pyplot(plot_1_passengers_survivability_gender())
    elif selected_option[0] == "Age bar graph":
        st.pyplot(plot_2_non_adults_survivability())
    elif selected_option[0] == "Class bar graph":
        st.pyplot(plot_3_passangers_class_survivability())
    elif selected_option[0] == "Age histogram":
        st.pyplot(plot_4_age_survivalibity_histograms())
    elif selected_option[0] == "Fare histogram":
        st.pyplot(plot_5_fare_survivability_histograms())
    elif selected_option[0] == "Gender percentage":
        st.pyplot(plot_6_gender_survival_chance())
    elif selected_option[0] == "Mean fare by Class":
        st.pyplot(plot_7_mean_fare_class())
    elif selected_option[0] == "Survival due to fare":
        st.pyplot(plot_8_fare_and_survival_scatterplots())


def left_panel_tools() -> st.sidebar.radio:
    """
    Function displays sidebar and provides tools for data analysis.

    Parameters:
     - None.

    Returns:
     - selected_option, main option - st.sidebar.radio
    """
    st.sidebar.title("Options")
    st.sidebar.title("Factors selection")

    main_option = st.sidebar.radio(
        "Data select: ",
        ["Passengers count", "Survivability", "Fare"],
        key=["a", "b", "c"],
    )

    if main_option == "Passengers count":
        fig_number = list(range(1, 4))
        selected_option = st.sidebar.radio(
            "Factor select: ",
            ["Gender and Class"],
            key=[f"Wykres{i}" for i in fig_number],
        )
        # return selected_option
    elif main_option == "Survivability":
        fig_number = list(range(1, 6))
        selected_option = st.sidebar.radio(
            "Factor select:",
            [
                "Gender bar graph",
                "Age bar graph",
                "Class bar graph",
                "Age histogram",
                "Fare histogram",
                "Gender percentage",
            ],
            key=[f"Wykres{i}" for i in fig_number],
        )
        # return selected_option
    elif main_option == "Fare":
        fig_number = list(range(1, 3))
        selected_option = st.sidebar.radio(
            "Select factor:",
            ["Mean fare by Class", "Survival due to fare"],
            key=[f"Wykres{i}" for i in fig_number],
        )

    return selected_option, main_option


def searchable_table():
    """
    Function displays searchable table for titanic data in sidebar and
    displays searched data on mainpage.

    Parameters:
     - None.

    Returns:
     - None.
    """

    df_titanic = titanic_data()

    search_data = ["Name", "PassengerId", "Age", "Fare"]
    select_data = st.sidebar.radio("Select datatype to search", search_data)
    st.divider()
    st.write(select_data)

    search_term = None

    if select_data == "Name":
        search_term = st.sidebar.text_input("Search Name:", "")
        st.sidebar.dataframe(df_titanic["Name"])
        filtered_df = df_titanic[
            df_titanic[select_data].str.contains(search_term, case=False)
        ]

    elif select_data == "PassengerId":
        search_term = st.sidebar.text_input("Search Passenger by ID:", "")
        passenger_ids = [
            int(x.strip()) for x in search_term.split(",") if x.strip().isdigit()
        ]

        if passenger_ids:
            st.sidebar.dataframe(
                df_titanic[df_titanic[select_data].isin(passenger_ids)]
            )
            filtered_df = df_titanic[df_titanic[select_data].isin(passenger_ids)]
        else:
            st.sidebar.warning("Please enter Passenger IDs separated by commas.")

    elif select_data == "Age":
        age_range = st.sidebar.slider(
            "Select Age Range",
            float(df_titanic["Age"].min()),
            float(df_titanic["Age"].max()),
            (float(df_titanic["Age"].min()), float(df_titanic["Age"].max())),
        )
        st.sidebar.dataframe(df_titanic["Age"])
        filtered_df = df_titanic[
            (df_titanic[select_data] >= age_range[0])
            & (df_titanic[select_data] <= age_range[1])
        ]
        st.dataframe(filtered_df)

    elif select_data == "Fare":
        fare_range = st.sidebar.slider(
            "Select Fare Range",
            float(df_titanic["Fare"].min()),
            float(df_titanic["Fare"].max()),
            (float(df_titanic["Fare"].min()), float(df_titanic["Fare"].max())),
        )
        st.sidebar.dataframe(df_titanic["Fare"])
        filtered_df = df_titanic[
            (df_titanic[select_data] >= fare_range[0])
            & (df_titanic[select_data] <= fare_range[1])
        ]
        st.dataframe(filtered_df)

    if search_term:
        st.dataframe(filtered_df)


def show_mainpage():
    """
    Function displays page.
    """
    selected_option = left_panel_tools()
    header(selected_option)
    main_page(selected_option)
    searchable_table()


if __name__ == "__main__":
    #  streamlit run d:/_NEWHOPE/Dziadek_03/TestRepo/src/task_7_data_visual.py
    show_mainpage()
