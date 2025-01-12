import pandas as pd
from ressources.reader_csv import load_data
import numpy as np


def visualisation_donnees_manquantes(dataframe):
    """
    Affiche les colonnes avec des données manquantes et leur pourcentage dans la DataFrame.
    - Supprime automatiquement les colonnes ayant 50% ou plus de valeurs manquantes.

    Arguments :
    dataframe : pd.DataFrame : La DataFrame à analyser.

    Retourne :
    Une DataFrame mise à jour après la suppression des colonnes avec trop de valeurs manquantes.
    """
    # Collecte des colonnes qui contiennent des valeurs manquantes
    col_with_na = dataframe.columns[dataframe.isna().any()].tolist()

    # Calcul des statistiques de valeurs manquantes (nombre et pourcentage)
    stats = [
        (col, dataframe[col].isna().sum(), round(dataframe[col].isna().sum() / len(dataframe) * 100, 2))
        for col in col_with_na
    ]

    # Affichage des statistiques des colonnes avec des valeurs manquantes
    if stats:
        for col, nb_na, perc_na in stats:
            if perc_na <= 50.0:
                print(f"🔹 **Colonne:** {col} | **Données manquantes:** {nb_na} | **Pourcentage:** {perc_na}%")

        # Identification des colonnes ayant 50% ou plus de valeurs manquantes
        cols_to_drop = [col for col, _, perc_na in stats if perc_na >= 50]

        # Suppression automatique des colonnes identifiées
        if cols_to_drop:
            dataframe = dataframe.drop(columns=cols_to_drop)

    return dataframe


def processing():
    """
    Effectue le pré-traitement des données :
    - Chargement des données
    - Suppression des colonnes inutiles
    - Remplacement de certaines valeurs par NaN

    Retourne :
    DataFrame pré-traitée
    """
    data = load_data()  # Chargement des données (fonction supposée définie ailleurs)
    pd.options.mode.chained_assignment = None  # Désactivation des avertissements liés aux copies de DataFrame

    # Affichage des informations de la DataFrame avant traitement
    print("DataFrame Initiale".center(50, "="))
    data.info()
    print()
    print("Taux de valeurs manquantes".center(50, "="))

    # Suppression des colonnes avec plus de 50% de valeurs manquantes
    new_data = visualisation_donnees_manquantes(data)

    # Liste des colonnes spécifiques à supprimer (jugées non pertinentes)
    colonnes_a_supprimer = [
        "Diplôme", "Disciplines", "Salaire net mensuel médian national",
        "Salaire net mensuel médian des emplois à temps plein", "Discipline",
        "Secteur disciplinaire", "Code du domaine", "Secteur disciplinaire",
        "Code du secteur disciplinaire", "Code du secteur disciplinaire SISE",
        "Nombre de réponses", "Part des emplois de niveau cadre ou profession intermédiaire",
        "% emplois extérieurs à la région de l’université", "Code de la discipline",
        "Premier quartile des salaires nets mensuels des emplois à temps plein",
        "Dernier quartile des salaires nets mensuels des emplois à temps plein", "cle_DISC",
        "Salaire net mensuel national 1er quartile", "Salaire net mensuel national 3ème quartile",
    ]

    # Suppression des colonnes spécifiées
    new_data = new_data.drop(columns=colonnes_a_supprimer)

    # Remplacement des valeurs "nd" et "ns" par NaN
    new_data.replace(["nd", "ns"], np.nan, inplace=True)

    return new_data


def supprimer_replace_nan(dataframe):
    """
    Effectue des conversions de types et supprime les valeurs manquantes.
    - Convertit certaines colonnes en nombres flottants.
    - Convertit certaines colonnes en chaînes de caractères.
    - Supprime les lignes contenant des valeurs manquantes.

    Arguments :
    dataframe : pd.DataFrame : La DataFrame à traiter.

    Retourne :
    DataFrame nettoyée sans valeurs manquantes.
    """
    # Liste des colonnes à convertir en type float (numérique)
    list_colonne_conv_float = [
        "Taux d’insertion", "Part des emplois de niveau cadre",
        "Part des emplois à temps plein", "Salaire brut annuel estimé",
        "Part des diplômés boursiers dans la discipline", "Part des emplois stables"
    ]

    # Liste des colonnes à convertir en type string (texte)
    list_colonne_conv_str = ["Année", "situation", "Genre", "Domaine"]

    # Conversion des colonnes numériques
    for colonne in list_colonne_conv_float:
        if colonne in dataframe.columns:
            dataframe[colonne] = pd.to_numeric(dataframe[colonne], errors="coerce")

    # Conversion des colonnes textuelles
    for colonne in list_colonne_conv_str:
        if colonne in dataframe.columns:
            dataframe[colonne] = dataframe[colonne].astype(str)

    # Suppression des lignes contenant des valeurs manquantes
    dataframe = dataframe.dropna()

    return dataframe
