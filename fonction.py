import pandas as pd
from ressources.reader_csv import load_data
import numpy as np

def visualisation_donnees_manquantes2(dataframe):
    """
    Affiche les colonnes avec des données manquantes et leur pourcentage dans la DataFrame.
    - Supprime automatiquement les colonnes ayant 50% ou plus de valeurs manquantes.
    - Demande à l'utilisateur s'il souhaite supprimer les lignes contenant des valeurs manquantes pour les autres colonnes.

    Arguments :
    dataframe : pd.DataFrame : La DataFrame à analyser.

    Retourne :
    Une DataFrame mise à jour après la suppression des colonnes et lignes choisies par l'utilisateur.
    """

    print("Résultats statistiques valeurs manquantes".center(50, "="))

    # Collecte des colonnes avec des données manquantes
    col_with_na = dataframe.columns[dataframe.isna().any()].tolist()

    # Calcul des statistiques de valeurs manquantes
    stats = [
        (col, dataframe[col].isna().sum(), round(dataframe[col].isna().sum() / len(dataframe) * 100, 2))
        for col in col_with_na
    ]

    # Affichage des statistiques
    if stats:
        for col, nb_na, perc_na in stats:
            print(f"🔹 **Colonne:** {col} | **Données manquantes:** {nb_na} | **Pourcentage:** {perc_na}%")

        # Suppression automatique des colonnes avec ≥ 50% de valeurs manquantes
        cols_to_drop = [col for col, _, perc_na in stats if perc_na >= 50]

        if cols_to_drop:
            dataframe = dataframe.drop(columns=cols_to_drop)

        # Mise à jour des colonnes restantes avec des valeurs manquantes
        col_with_na = [col for col, _, perc_na in stats if perc_na < 50]

        # Interaction utilisateur pour suppression des lignes
        cols_to_clean = [
            col for col in col_with_na if input(f"🗑 **Souhaitez-vous supprimer les lignes avec des valeurs manquantes dans '{col}' ? (Y/N) : **").strip().lower() == "y"
        ]

        if cols_to_clean:
            dataframe = dataframe.dropna(subset=cols_to_clean)

        else:
            print("✅ **Aucune suppression de ligne effectuée.**")

    else:
        print("✅ **Aucune colonne avec des valeurs manquantes.**")

    return dataframe


def visualisation_donnees_manquantes(dataframe):
    """
    Affiche les colonnes avec des données manquantes et leur pourcentage dans la DataFrame.
    - Supprime automatiquement les colonnes ayant 50% ou plus de valeurs manquantes.
    - Demande à l'utilisateur s'il souhaite supprimer les lignes contenant des valeurs manquantes pour les autres colonnes.

    Arguments :
    dataframe : pd.DataFrame : La DataFrame à analyser.

    Retourne :
    Une DataFrame mise à jour après la suppression des colonnes et lignes choisies par l'utilisateur.
    """
    # Collecte des colonnes avec des données manquantes
    col_with_na = dataframe.columns[dataframe.isna().any()].tolist()

    # Calcul des statistiques de valeurs manquantes
    stats = [
        (col, dataframe[col].isna().sum(), round(dataframe[col].isna().sum() / len(dataframe) * 100, 2))
        for col in col_with_na
    ]

    # Affichage des statistiques
    if stats:
        for col, nb_na, perc_na in stats:
            if perc_na <= 50.0:
                print(f"🔹 **Colonne:** {col} | **Données manquantes:** {nb_na} | **Pourcentage:** {perc_na}%")

        # Suppression automatique des colonnes avec ≥ 50% de valeurs manquantes
        cols_to_drop = [col for col, _, perc_na in stats if perc_na >= 50]

        if cols_to_drop:
            dataframe = dataframe.drop(columns=cols_to_drop)

    return dataframe


def processing():
    data = load_data()
    pd.options.mode.chained_assignment = None
    # Afficher information dataset
    print("DataFrame Initiale".center(50, "="))
    data.info()

    #Afficher stats valeurs manquantes
    print()
    print("Taux de valeurs manquantes".center(50, "="))
    new_data = visualisation_donnees_manquantes(data)
    # print(new_data.info())
    colonnes_a_supprimer = ["Diplôme", "Disciplines", "Salaire net mensuel médian national",
                            "Salaire net mensuel médian des emplois à temps plein", "Discipline",
                            "Secteur disciplinaire", "Code du domaine", "Secteur disciplinaire",
                            "Code du secteur disciplinaire", "Code du secteur disciplinaire SISE",
                            "Nombre de réponses", "Part des emplois de niveau cadre ou profession intermédiaire",
                            "% emplois extérieurs à la région de l’université","Code de la discipline",
                            "Premier quartile des salaires nets mensuels des emplois à temps plein",
                            "Dernier quartile des salaires nets mensuels des emplois à temps plein", "cle_DISC",
                            "Salaire net mensuel national 1er quartile", "Salaire net mensuel national 3ème quartile",
                            ]

    new_data = new_data.drop(columns=colonnes_a_supprimer)

    # remplacer nd et ns en nan
    new_data.replace(['nd', 'ns'], np.nan, inplace=True)

    return new_data


def supprimer_replace_nan(dataframe):

    # Définition des colonnes à convertir
    list_colonne_conv_float = [
        "Taux d’insertion", "Part des emplois de niveau cadre",
        "Part des emplois à temps plein", "Salaire brut annuel estimé",
        "Part des diplômés boursiers dans la discipline", 'Part des emplois stables'
    ]  # "Part des emplois stables"""

    list_colonne_conv_str = ["Année", "situation", "Genre", "Domaine"]

    for colonne in list_colonne_conv_float:
        if colonne in dataframe.columns:
            dataframe[colonne] = pd.to_numeric(dataframe[colonne], errors="coerce")

    for colonne in list_colonne_conv_str:
        if colonne in dataframe.columns:
            dataframe[colonne] = dataframe[colonne].astype(str)

    # Remplacement des valeurs manquantes
    # dataframe.fillna(dataframe.mean(numeric_only=True), inplace=True)
    # dataframe.fillna(dataframe.median(numeric_only=True), inplace=True)
    dataframe = dataframe.dropna()

    return dataframe
