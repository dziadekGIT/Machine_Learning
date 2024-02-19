import streamlit as st
from lesson_plots import (
    sinus_plot,
    scatterplot_iris,
    barplot_iris,
    histogram_iris,
    load_iris_data,
)


def run_streamlit():
    """
    streamlit run
    """
    df = load_iris_data()

    example_value = st.slider("What will happen?", 0, 130, 25)
    st.write("value is:", example_value)

    number = 50
    number = st.number_input(
        "Sinus function max", min_value=0, max_value=100, value=number
    )
    st.write("The current number is ", number)

    st.header(f"Sinus Function Plot of {number}")
    number = 50 if number is None else number
    st.pyplot(sinus_plot(number))

    st.header("Scatter Plot")
    st.pyplot(scatterplot_iris(df))

    st.header("Bar Plot")
    st.pyplot(barplot_iris(df))

    st.header("Histogram")
    st.pyplot(histogram_iris(df))


if __name__ == "__main__":
    # streamlit run lesson_streamlit.py
    run_streamlit()
