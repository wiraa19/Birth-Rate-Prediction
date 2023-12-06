import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def plot_weekday_distribution(data):
    plt.figure(figsize=(15, 4))
    sns.set_style("darkgrid", {"axes.facecolor": "0.9", 'grid.color': '.6', 'grid.linestyle': '-.'})
    dist1 = data.groupby("day_of_week")["births"].mean()

    dist1.plot(kind='bar', rot=0)
    plt.xlabel("day_of_week", fontsize=14, color="r")
    plt.ylabel("births", fontsize=14, color="r")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Weekdays' distribution", fontsize=16, color="r")

    # Return the Matplotlib figure
    return plt.gcf()

def train_linear_model(data):
    X = data[['year', 'month', 'date_of_month', 'day_of_week']]
    y = data['births']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test

def main():
    st.title("Birth Rate Prediction App")

    # Memuat data
    file = pd.read_csv('C:\\Users\\user\\Desktop\\birth date prediction\\US_births_2000-2014_SSA.csv')

    # Memanggil fungsi visualisasi dan menampilkannya menggunakan st.pyplot()
    fig = plot_weekday_distribution(file)
    st.pyplot(fig)

    # Training model
    model, X_test, y_test = train_linear_model(file)

    # Prediksi
    st.header("Prediction")
    year = st.slider("Select Year", min_value=int(file['year'].min()), max_value=int(file['year'].max()), step=1)
    month = st.slider("Select Month", min_value=int(file['month'].min()), max_value=int(file['month'].max()), step=1)
    day_of_month = st.slider("Select Day of Month", min_value=int(file['date_of_month'].min()), max_value=int(file['date_of_month'].max()), step=1)
    day_of_week = st.slider("Select Day of Week", min_value=int(file['day_of_week'].min()), max_value=int(file['day_of_week'].max()), step=1)

    prediction_input = np.array([[year, month, day_of_month, day_of_week]])
    prediction = model.predict(prediction_input)[0]

    st.write(f"Predicted Births: {prediction:.2f}")

# Menjalankan aplikasi Streamlit
if __name__ == "__main__":
    main()

