import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pandas import DataFrame
from typing import List
from matplotlib import pyplot as plt, patches
from fonction import visualisation_donnees_manquantes
from ressources.reader_csv import *
from analyse_exploratoire import processing2
from statsmodels.regression.linear_model import RegressionResults


new_data = processing2()
print(new_data)

#supprimer colonnes qui ne servent à rien
colonnes_a_supprimer = ["Diplôme","Disciplines","Salaire net mensuel médian national","Salaire net mensuel médian des emplois à temps plein", "Discipline","Secteur disciplinaire","Nombre de réponses","Part des emplois de niveau cadre ou profession intermédiaire","% emplois extérieurs à la région de l’université", "Premier quartile des salaires nets mensuels des emplois à temps plein","Dernier quartile des salaires nets mensuels des emplois à temps plein"]
data = new_data.drop(columns=colonnes_a_supprimer)

#voir contenu unique des colonnes pour détecter nan, ns, nd
for col in new_data.columns:
    print(f"\nColonne : {col}")
    print("Valeurs uniques :", new_data[col].unique())

#remplacer nd et ns en nan
data.replace(['nd', 'ns'], np.nan, inplace=True)

# Compter les valeurs manquantes pour chaque colonne
missing_values_count = data.isnull().sum()

# Afficher les résultats
print(missing_values_count)

# Supprimer les lignes avec des valeurs manquantes
data_cleaned = data.dropna()

# Vérifier le DataFrame nettoyé
print(data_cleaned)

# Compter les valeurs manquantes pour chaque colonne
missing_values_count = data_cleaned.isnull().sum()
print(missing_values_count)

data1= data_cleaned


#encoder les variables quantitatives
data_cleaned = pd.get_dummies(data_cleaned, columns=['Genre', 'Domaine', 'situation'], drop_first=True)

# encoder et normaliser valeurs numériques
from sklearn.preprocessing import StandardScaler

numerical_cols = [
    "Année","Taux d’insertion",'Part des emplois de niveau cadre', 'Part des emplois stables',
    'Part des emplois à temps plein', 'Salaire brut annuel estimé',
    'Part des diplômés boursiers dans la discipline', 'Taux de chômage national'
]

scaler = StandardScaler()
data_cleaned[numerical_cols] = scaler.fit_transform(data_cleaned[numerical_cols])



# Pairplot1 Colonnes numériques
numeric_cols1 = [
    "Taux d’insertion", 'Salaire brut annuel estimé',
    'Part des diplômés boursiers dans la discipline', 'Taux de chômage national'
]

# Pairplot1 pour distinguer les groupes
sns.set_context("notebook", font_scale=0.6)
sns.pairplot(data_cleaned[numeric_cols1])
#Modifier la taille des textes sur l'axe Y pour chaque sous-graphique

plt.show()

# Pairplot2 Colonnes numériques
numeric_cols2 = [
    "Taux d’insertion", "Part des emplois de niveau cadre", "Part des emplois stables",
    "Part des emplois à temps plein"
]

# Pairplot2 pour distinguer les groupes
sns.set_context("notebook", font_scale=0.6)
sns.pairplot(data_cleaned[numeric_cols2])
# Modifier la taille des textes sur l'axe Y pour chaque sous-graphique
plt.show()



#from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

X = data_cleaned.drop(columns=['Taux d’insertion'])
y = data_cleaned['Taux d’insertion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Évaluez le modèle
print("R² :", r2_score(y_test, y_pred))
print("Erreur quadratique moyenne :", mean_squared_error(y_test, y_pred))

coefficients = pd.DataFrame({
    'Variable': X_train.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print(coefficients)

plt.scatter(y_test, y_pred)
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Comparaison des valeurs réelles et prédites")
plt.show()


import statsmodels.api as sm

#densité résidus
residuals = y_test - y_pred
# Tracer la distribution des résidus
plt.figure(figsize=(12, 8))
sns.kdeplot(residuals, fill=True, color="blue", alpha=0.6, label="Résidus")
plt.axvline(np.mean(residuals), color="red", linestyle="--", label="Moyenne des résidus")

# Ajouter une courbe normale
sns.kdeplot(
    np.random.normal(np.mean(residuals), np.std(residuals), len(residuals)),
    color="green", linestyle="--", label="Distribution normale"
)

# Ajouter les légendes et le titre
plt.title("Distribution des Résidus", fontweight="bold")
plt.legend()
plt.grid(True)
plt.show()

#homoscédacité
# Tracer les résidus vs les valeurs prédites
plt.figure(figsize=(12, 8))
plt.scatter(y_pred, residuals, alpha=0.6, color="blue")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Valeurs prédites")
plt.ylabel("Résidus")
plt.title("Valeurs prédites vs Résidus")
plt.grid(True)
plt.show()

print(X_test.head(10))  # Afficher les 10 premières lignes

#changer variables booléennes en numérique
#id les variables bool
bool_columns = X_test.select_dtypes(include=['bool']).columns
print("Colonnes booléennes :", bool_columns)
#changer bool en num
X_test[bool_columns] = X_test[bool_columns].astype(int)
#verifier le type des variables
bool_columns = X_test.select_dtypes(include=['bool']).columns
print("Colonnes booléennes :", bool_columns)

from statsmodels.stats.diagnostic import het_goldfeldquandt
import statsmodels.api as sm
import numpy as np

# Étape 1 : Calculer les résidus
residuals = y_test - y_pred  # y_test : valeurs réelles, y_pred : valeurs prédites

# Étape 2 : Ajouter une constante à X_test (si nécessaire)
X_test_with_const = sm.add_constant(X_test)

# Étape 3 : Effectuer le test de Goldfeld-Quandt
test_stat, p_value, _ = het_goldfeldquandt(residuals, X_test_with_const)

# Étape 4 : Afficher les résultats
print("Statistique du test Goldfeld-Quandt :", test_stat)
print("p-value :", p_value)

if p_value > 0.05:
    print("Aucune preuve d'hétéroscédasticité (p > 0.05)")
else:
    print("Présence d'hétéroscédasticité (p <= 0.05)")

import statsmodels.api as sm
import matplotlib.pyplot as plt

# Tracer la fonction d'autocorrélation des résidus
sm.graphics.tsa.plot_acf(residuals, lags=40)
plt.title("Autocorrélation des Résidus")
plt.xlabel("Lags")
plt.ylabel("Autocorrélation")
plt.grid(True)
plt.show()


# 5 -Calculer la matrice de corrélation
correlation_matrix = X_train.corr()

# Afficher la matrice de corrélation sous forme de heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de Corrélation des Variables Explicatives", fontweight="bold")
plt.show()

# Identifier les paires de variables fortement corrélées
threshold = 0.8
high_corr = correlation_matrix[(correlation_matrix > threshold) | (correlation_matrix < -threshold)].stack()
high_corr = high_corr[high_corr.index.get_level_values(0) != high_corr.index.get_level_values(1)].drop_duplicates()
print("Variables fortement corrélées :")
print(high_corr)

file_path = "C:\\Users\\ronan\\Desktop\\Projet Stats\\data1.csv"
data1.to_csv(file_path, index=False)
print(f"Le fichier a été exporté avec succès : {file_path}")
