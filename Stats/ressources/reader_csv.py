import pandas
from pandas import DataFrame


def load_data():
    data = pandas.read_csv("data-insertion_pro.csv", delimiter=";")
    print(data.head())  # Display the first few rows of the DataFrame
    print(data.columns)
    print(data.shape)
    return data

