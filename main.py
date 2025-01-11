import pandas as pd
from ressources.reader_csv import *
from fonction import visualisation_donnees_manquantes

dataframe = load_data()
print(dataframe.head())  # Display the first few rows of the DataFrame
print(dataframe.columns)
print(dataframe.shape)
print(dataframe.info())
new_data = visualisation_donnees_manquantes(dataframe)
print(new_data.info())
print(new_data.columns)

columns_with_missing_values = new_data.columns[new_data.isnull().any()].tolist()

print("Colonnes avec des valeurs manquantes :", columns_with_missing_values)

# Liste des colonnes spécifiques pour lesquelles vous voulez calculer la moyenne
colonnes_cibles = ['Salaire net mensuel médian des emplois à temps plein', 'Salaire brut annuel estimé','Premier quartile des salaires nets mensuels des emplois à temps plein','Dernier quartile des salaires nets mensuels des emplois à temps plein','Salaire net mensuel national 1er quartile', 'Salaire net mensuel national 3ème quartile']


# Remplacer les valeurs manquantes par la moyenne de chaque colonne
for colonne in colonnes_cibles:
    # Calculer la moyenne en ignorant les NaN
    moyenne = new_data[colonne].mean(skipna=True)

    # Remplacer les NaN par la moyenne
    new_data[colonne].fillna(moyenne, inplace=True)

# Afficher les premières lignes pour vérifier
moyennes_colonnes1 = new_data[colonnes_cibles].mean()
print("Moyennes des colonnes spécifiques :")
print(moyennes_colonnes1)


