import pandas as pd
import numpy as np
import seaborn as sns
from pandas import DataFrame
from typing import List
from matplotlib import pyplot as plt, patches
from fonction import visualisation_donnees_manquantes
from ressources.reader_csv import load_data

def processing():
    data = load_data()

    # Afficher information dataset
    print("Informations sur le dataset".center(50, "="))
    print(data.info())

    #Afficher stats valeurs manquantes
    new_data = visualisation_donnees_manquantes(data)
    print(new_data.info())

    data = new_data.drop(columns=[col for col in new_data.columns if "Salaire net" in col])
    print("\nColonnes contenant 'Salaire net' supprimées.")
    print("Colonnes restantes :")
    print(data.info())

    data = new_data.drop(columns=[col for col in new_data.columns if "Code" in col])
    print("\nColonnes contenant 'Code' supprimées.")
    print("Colonnes restantes :")
    print(data.info())

    data = data.drop(columns=["cle_DISC","Salaire net mensuel national 1er quartile","Salaire net mensuel national 3ème quartile"])
    print(data.info())

    #Variables Quantitatives
    # Colonnes à convertir
    var_quantitative = data[['Taux d’insertion', 'Part des emplois de niveau cadre ou profession intermédiaire', 'Part des emplois de niveau cadre', 'Part des emplois stables','Part des emplois à temps plein','Salaire net mensuel médian des emplois à temps plein','Part des diplômés boursiers dans la discipline']]

    # Convertir les colonnes quantitatives en numérique
    for col in var_quantitative.columns:
        var_quantitative[col] = pd.to_numeric(var_quantitative[col], errors='coerce')  # Convertir en numérique, NaN pour les erreurs

    # Afficher les données après conversion
    print(var_quantitative.info())
    print(var_quantitative)

    # Calculer la matrice de corrélation pour les colonnes numériques uniquement
    corr_matrix = var_quantitative.select_dtypes(include=[np.number]).corr()

    # Afficher la matrice de corrélation
    print("\nMatrice de corrélation :")
    print(corr_matrix)

    # Créer une heatmap des corrélations
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Heatmap des corrélations')
    plt.show()

    #Variables qualitatives
    print(data.dtypes)
    # Convertir 'Taux d’insertion' en numérique
    data["Taux d’insertion"] = pd.to_numeric(data["Taux d’insertion"], errors='coerce')

    # Supprimer les lignes avec des NaN dans 'Taux d’insertion' ou 'Genre'
    data = data.dropna(subset=["Taux d’insertion", "Diplôme"])

    # Vérifiez les types après conversion
    print("\nTypes de données après conversion :")
    print(data.dtypes)

    # Tracer le KDE plot
    sns.kdeplot(data=data, x="Taux d’insertion", hue="Diplôme", fill=True, alpha=0.6)
    plt.title("Effet des modalités de 'Genre' sur 'Taux d’insertion'")
    plt.show()
    return data

def processing2():
    data = load_data()

    # Afficher information dataset
    print("Informations sur le dataset".center(50, "="))
    print(data.info())

    #Afficher stats valeurs manquantes
    new_data = visualisation_donnees_manquantes(data)
    print(new_data.info())

    data = new_data.drop(columns=[col for col in new_data.columns if "Salaire net" in col])
    print("\nColonnes contenant 'Salaire net' supprimées.")
    print("Colonnes restantes :")
    print(data.info())

    data = new_data.drop(columns=[col for col in new_data.columns if "Code" in col])
    print("\nColonnes contenant 'Code' supprimées.")
    print("Colonnes restantes :")
    print(data.info())

    data = data.drop(columns=["cle_DISC","Salaire net mensuel national 1er quartile","Salaire net mensuel national 3ème quartile"])


    return data

if __name__ == "__main__":
    new_data = processing()
    print(new_data)


