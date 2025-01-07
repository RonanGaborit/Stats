from ressources.reader_csv import *
from fonction import *


print(verifcation())
dataframe = load_data()
print(dataframe.head())  # Display the first few rows of the DataFrame
print(dataframe.columns)
print(dataframe.shape)
print(dataframe.info())

new_data = visualisation_donnees_manquantes(dataframe)
print(new_data.shape)
print(new_data.columns)

