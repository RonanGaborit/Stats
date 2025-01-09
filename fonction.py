import pandas as pd
from typing import Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, patches
from pandas import DataFrame


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


class GlobalAnalysis:
    """
    This class contains methods to perform global analysis on the data (missing values, infos, ...)
    """

    @staticmethod
    def print_nan_statistics(data: DataFrame) -> None:
        print(" Missing values in the data ".center(50, "="))
        print(data.isna().sum())

    @staticmethod
    def print_info(data: DataFrame) -> None:
        print(" Information about the data ".center(50, "="))
        print(data.info())


class QuantitativeAnalysis:
    """
    This class contains methods to perform analysis on the quantitative data (describe, correlation, ...)
    """

    @staticmethod
    def print_describe(data: DataFrame) -> None:
        print(" Descriptive statistics of the data ".center(50, "="))
        print(data.describe())

    @staticmethod
    def plot_linear_correlation(data: DataFrame, colors: list[str]) -> None:
        corr = data.corr(method="pearson")
        mask = np.triu(np.ones_like(corr, dtype=bool))

        plt.figure(figsize=(12, 8))
        plt.title(
            label="Coefficient de corrélation linéaire entre les variables",
            fontsize=13,
            fontweight="bold",
        )
        sns.heatmap(corr, annot=True, cmap="mako", mask=mask, square=True, alpha=0.6)

        # adding rectangle
        ax = plt.gca()

        rect = patches.Rectangle(
            (0, 0), 1, data.shape[0], linewidth=1, edgecolor=colors[1], facecolor="none"
        )
        ax.add_patch(rect)

        plt.show()

    @staticmethod
    def plot_pairplot(
        data: DataFrame, colors: list[str], hue: Optional[str] = None
    ) -> None:
        if hue:
            n_modalities = data.loc[:, hue].nunique()
            sns.pairplot(
                data,
                plot_kws={"alpha": 0.6},
                diag_kws={"fill": True, "alpha": 0.6},
                diag_kind="kde",
                kind="scatter",
                palette=colors[:n_modalities],
                hue=hue,
            )
        else:
            sns.pairplot(
                data,
                plot_kws={"alpha": 0.6, "color": colors[1]},
                diag_kws={"fill": True, "alpha": 0.6, "color": colors[0]},
                diag_kind="kde",
                kind="scatter",
            )
        plt.show()


class QualitativeAnalysis:
    """
    This class contains methods to perform analysis on the qualitative data
    """

    @staticmethod
    def print_modalities_number(data: DataFrame) -> None:
        print(data.nunique(axis=0))

    @staticmethod
    def plot_modalities_effect_on_target(
        data: DataFrame, target_column: str, qualitative_column: str, colors: list[str]
    ) -> None:

        palette = sns.color_palette(
            palette=colors, n_colors=data.loc[:, qualitative_column].nunique()
        )

        plt.figure(figsize=(12, 8))

        sns.kdeplot(
            data=data,
            x=target_column,
            hue=qualitative_column,
            palette=palette,
            fill=True,
            alpha=0.6,
        )
        for modality in data.loc[:, qualitative_column].unique():
            plt.axvline(
                x=data.loc[
                    data.loc[:, qualitative_column] == modality, target_column
                ].mean(),
                color=palette[
                    data.loc[:, qualitative_column].unique().tolist().index(modality)
                ],
                linestyle="--",
                label=f"{modality} mean",
            )

        plt.title(
            label=f"Effet des modalités de la variable {qualitative_column} sur la variable cible ({target_column})",
            fontsize=13,
            fontweight="bold",
        )
        plt.ylabel("Densité")
        plt.grid(True)
        plt.show()
