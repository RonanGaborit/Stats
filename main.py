from ressources.reader_csv import *
from fonction import *


print(verifcation())
dataframe = load_data()
print(dataframe.info())

new_data = visualisation_donnees_manquantes(dataframe)

