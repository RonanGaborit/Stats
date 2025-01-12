import plotly.express as px
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pre_processing import data_clean


# Charger les données avec mise en cache
@st.cache_data
def load_data():
    data = data_clean()
    data.columns = data.columns.str.strip().str.replace('’', "'", regex=False)
    return data

# Charger les données
data = load_data()

# Créer un sélecteur pour choisir une page
page = st.sidebar.selectbox("Naviguer vers :", ["Statistiques et Corrélations", "Exploration"])

# **Page : Statistiques et Corrélations**
if page == "Statistiques et Corrélations":
    st.title("Statistiques et Corrélations")

    # Aperçu des données
    st.header("Aperçu des données")
    st.dataframe(data)

    # Statistiques descriptives
    st.header("Statistiques descriptives")
    columns_to_describe = data.select_dtypes(include=['number']).drop(columns=['Année'], errors='ignore')
    st.write(columns_to_describe.describe())

    # Distribution du taux d'insertion
    st.header("Distribution du Taux d'Insertion")
    fig, ax = plt.subplots()
    sns.histplot(data["Taux d'insertion"], kde=True, ax=ax)
    ax.set_title("Distribution du Taux d'Insertion")
    st.pyplot(fig)

    # Matrice de corrélation
    st.header("Matrice de Corrélation")
    data_matrix_cor = pd.get_dummies(data)
    correlation_matrix = data_matrix_cor.corr()
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Corrélations entre les Variables Numériques")
    st.pyplot(fig)

    # Relations avec le Taux d'Insertion
    st.header("Relations avec le Taux d'Insertion")
    variables = [
        "Part des emplois de niveau cadre",
        "Part des emplois stables",
        "Part des emplois à temps plein",
        "Salaire brut annuel estimé",
        "Part des diplômés boursiers dans la discipline",
    ]

    for var in variables:
        fig, ax = plt.subplots()
        sns.scatterplot(x=data[var], y=data["Taux d'insertion"], ax=ax)
        ax.set_title(f"Taux d'Insertion vs {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Taux d'insertion")
        st.pyplot(fig)

    # Box Plot : Taux de Chômage National vs Taux d'Insertion
    st.header("Relation entre Taux de Chômage National et Taux d'Insertion")
    data["Taux de chômage (catégories)"] = pd.cut(
        data["Taux de chômage national"],
        bins=[0, 5, 10, 15, 20, 25],
        labels=["0-5%", "5-10%", "10-15%", "15-20%", "20-25%"],
        include_lowest=True,
    )
    fig = px.box(
        data,
        x="Taux de chômage (catégories)",
        y="Taux d'insertion",
        color="Taux de chômage (catégories)",
        title="Distribution du Taux d'Insertion par Catégories de Taux de Chômage",
        labels={"Taux de chômage (catégories)": "Taux de Chômage National (%)", "Taux d'insertion": "Taux d'Insertion (%)"},
        template="plotly_white",
    )
    st.plotly_chart(fig)

    # Box Plot : Taux d'Insertion par Année
    st.header("Taux d'Insertion par Année")
    fig1 = px.box(
        data,
        x="Année",
        y="Taux d'insertion",
        title="Distribution du Taux d'Insertion par Année",
        labels={"Taux d'insertion": "Taux d'Insertion (%)", "Année": "Année"},
        color="Année",
        template="plotly_white",
    )
    st.plotly_chart(fig1)

# **Page : Exploration**
elif page == "Exploration":
    st.title("Exploration des Données")

    # Filtres dans la barre latérale
    st.sidebar.header("Filtres")
    annee = st.sidebar.multiselect("Sélectionner les Années", options=data["Année"].unique(), default=data["Année"].unique())
    genre = st.sidebar.multiselect("Sélectionner le Genre", options=data["Genre"].unique(), default=data["Genre"].unique())
    domaine = st.sidebar.multiselect("Sélectionner le Domaine", options=data["Domaine"].unique(), default=data["Domaine"].unique())

    # Filtrage des données
    filtered_data = data[(data["Année"].isin(annee)) & (data["Genre"].isin(genre)) & (data["Domaine"].isin(domaine))]

    if filtered_data.empty:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés. Veuillez modifier les filtres.")
    else:
        # Graphique 1 :  Graphique avec la moyenne du Taux d'Insertion par Année et Genre
        filtered_data_grouped = filtered_data.groupby(["Année", "Genre"], as_index=False).agg(
            {"Taux d'insertion": "mean"}
        )
        st.subheader("Taux d'Insertion (Moyenne) par Année et Genre")
        print(filtered_data["Taux d'insertion"])
        fig1 = px.bar(
            filtered_data_grouped,
            x="Année",
            y="Taux d'insertion",
            color="Genre",
            barmode="group",
            title="Moyenne du Taux d'Insertion par Année et Genre",
            labels={"Taux d'insertion": "Taux d'Insertion (%)", "Année": "Année"}
        )
        st.plotly_chart(fig1)

        # Graphique 2 : Salaire Moyen par Domaine
        st.subheader("Salaire Moyen par Domaine")
        fig2 = px.box(
            filtered_data,
            x="Domaine",
            y="Salaire brut annuel estimé",
            color="Genre",
            title="Distribution des Salaires Brut Annuels par Domaine",
            labels={"Salaire brut annuel estimé": "Salaire Brut Annuel (€)"}
        )
        st.plotly_chart(fig2)


        #graphique 3
        st.header("Évolution du Taux de Chômage National")

        # Calculer la moyenne du taux de chômage par année
        chomage_par_annee = data.groupby('Année', as_index=False)['Taux de chômage national'].mean()

        # Exemple d'amélioration : Renommer les graphiques pour éviter les redondances
        # Graphique 1 : Évolution du Taux de Chômage National
        fig_chomage = px.line(
            chomage_par_annee,
            x='Année',
            y='Taux de chômage national',
            title="Évolution du Taux de Chômage National",
            labels={'Taux de chômage national': 'Taux de Chômage (%)', 'Année': 'Année'},
            markers=True
        )
        st.plotly_chart(fig_chomage)

        ## Créer un box plot interactif pour le Taux d'Insertion par Situation
        fig_insertion_situation = px.box(
        data,
        x='situation',
        y="Taux d'insertion",  # Utilisez le nom exact affiché dans data.columns
        color='situation',
        title="Taux d'Insertion par Situation",
        labels={'situation': 'Situation', "Taux d'insertion": "Taux d'Insertion (%)"},
        template="plotly_white"
        )

        # Afficher le graphique
        st.plotly_chart(fig_insertion_situation)

        # Données filtrées
        st.write("### Données Filtrées")
        st.dataframe(filtered_data)
