import pandas


path_data = "ressources\data-insertion_pro.csv"


def load_data():
    return pandas.read_csv(path_data, delimiter=";")

