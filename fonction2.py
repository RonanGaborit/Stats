from typing import Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, patches
from pandas import DataFrame

# from demos.utlis import setup_plot
from ressources.reader_csv import *
from fonction import *


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


if __name__ == "__main__":
    colors = setup_plot()
    data = load_data()
    # data = sns.load_dataset(name="mpg")

    quantitative_data = data.select_dtypes(include=np.number)
    qualitative_data = data.select_dtypes(include="object")

    # global analysis of the data
    GlobalAnalysis.print_info(data=data)
    GlobalAnalysis.print_nan_statistics(data=data)
    # only 6 missing values in the horsepower column
    # we can drop

    # analysis of the quantitative variables
    QuantitativeAnalysis.print_describe(data=quantitative_data)

    QuantitativeAnalysis.plot_linear_correlation(data=quantitative_data, colors=colors)
    # high correlation between the variables and target
    # multicolinearity between the variables

    QuantitativeAnalysis.plot_pairplot(data=data, colors=colors)
    # we see scale and (anti)correlation between the variables

    # analysis of the qualitative variables
    QualitativeAnalysis.print_modalities_number(data=qualitative_data)
    # name can be preprocessed more
    # however, since we have highly correlated variables, "name" can be dropped without losing to much informations

    QualitativeAnalysis.plot_modalities_effect_on_target(
        data=data, target_column="mpg", qualitative_column="origin", colors=colors
    )
    # we see that the origin seems to have an effect on the target variable
    # we can either encode it or ensure this effect through a central tendency statistical test (Student, Mann-Whitney, ...)
