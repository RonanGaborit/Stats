import pandas as pd


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

    print("\n------- Résultats --------\n")

    # Collecte des colonnes avec des données manquantes
    col_with_na = dataframe.columns[dataframe.isna().any()].tolist()

    # Calcul des statistiques de valeurs manquantes
    stats = [
        (col, dataframe[col].isna().sum(), round(dataframe[col].isna().sum() / len(dataframe) * 100, 2))
        for col in col_with_na
    ]

    # Affichage des statistiques
    if stats:
        print("📊 **Statistiques des valeurs manquantes**\n")
        for col, nb_na, perc_na in stats:
            if perc_na <= 50.0:
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


def verifcation():

    nom_auteur = ["DAOUDA", "Khadija"]
    return nom_auteur


def setup_plot():
    from aquarel import load_theme
    import seaborn as sns

    theme = load_theme("minimal_light")
    theme.apply()

    return ["#004aad", "#2bb4d4", "#2e2e2e", "#5ce1e6"]

