import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données

data = pd.read_csv('C:\\Users\\Sysai\\Downloads\\data1.csv', delimiter=',', encoding='UTF-8')
data.columns = data.columns.str.strip()

# Nettoyage des colonnes
data.columns = data.columns.str.strip().str.replace('’', "'", regex=False)


# Titre de l'application
st.title("Analyse du taux d'insertion des diplômés du supérieur")

# Aperçu des données
st.header("Aperçu des données")
st.dataframe(data)

# Statistiques descriptives
st.header("Statistiques descriptives")
st.write(data.describe())


# Distribution du taux d'insertion
st.header("Distribution du taux d'insertion")
fig, ax = plt.subplots()
sns.histplot(data["Taux d'insertion"], kde=True, ax=ax)
ax.set_title("Distribution du Taux d'insertion")
st.pyplot(fig)



# Matrice de corrélation
st.header("Matrice de corrélation")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)


# Scatter plots pour les variables les plus corrélées
st.header("Relations avec le Taux d'insertion")
variables = [
    "Part des emplois de niveau cadre",
    "Part des emplois stables",
    "Part des emplois à temps plein",
    "Salaire brut annuel estimé",
    "Part des diplômés boursiers dans la discipline",
    "Taux de chômage national"
]
for var in variables:
    fig, ax = plt.subplots()
    sns.scatterplot(x=data[var], y=data["Taux d'insertion"], ax=ax)
    ax.set_title(f"Taux d'insertion vs {var}")
    st.pyplot(fig)

# Modèle de régression linéaire
st.header("Modèle explicatif : Régression linéaire")
X = data[[
    "Part des emplois de niveau cadre",
    "Part des emplois stables",
    "Part des emplois à temps plein",
    "Salaire brut annuel estimé",
    "Part des diplômés boursiers dans la discipline",
    "Taux de chômage national"
]]
Y = data["Taux d'insertion"]

# Séparation des données
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = LinearRegression()
model.fit(X_train, Y_train)

# Prédictions
Y_pred = model.predict(X_test)

# Afficher les résultats
st.subheader("Performance du modèle")
st.write(f"MSE: {mean_squared_error(Y_test, Y_pred):.2f}")
st.write(f"R^2: {r2_score(Y_test, Y_pred):.2f}")

# Coefficients
st.subheader("Coefficients du modèle")
coefficients = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": model.coef_
})
st.write(coefficients)

st.write("Ce modèle permet de mieux comprendre les facteurs influençant le taux d'insertion des diplômés.")
