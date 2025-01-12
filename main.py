import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pre_processing import data_clean
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.diagnostic import het_goldfeldquandt
import statsmodels.api as sm

# Charger les données nettoyées depuis le module pre_processing
data_cleaned = data_clean()
data_cleaned.info()

# Encodage des variables catégorielles avec get_dummies (one-hot encoding)
data_cleaned = pd.get_dummies(data_cleaned, columns=['Genre', 'Domaine', 'situation'], drop_first=True)

# Normalisation des valeurs numériques
numerical_cols = [
    "Année", "Taux d’insertion", 'Part des emplois de niveau cadre', 'Part des emplois stables',
    'Part des emplois à temps plein', 'Salaire brut annuel estimé',
    'Part des diplômés boursiers dans la discipline', 'Taux de chômage national'
]
scaler = StandardScaler()
data_cleaned[numerical_cols] = scaler.fit_transform(data_cleaned[numerical_cols])

# Visualisation de la relation entre certaines variables avec un pairplot
numeric_cols1 = [
    "Taux d’insertion", 'Salaire brut annuel estimé',
    'Part des diplômés boursiers dans la discipline', 'Taux de chômage national'
]
sns.set_context("notebook", font_scale=0.6)
sns.pairplot(data_cleaned[numeric_cols1])
plt.show()

# Autre visualisation avec des variables supplémentaires
numeric_cols2 = [
    "Taux d’insertion", "Part des emplois de niveau cadre", "Part des emplois stables",
    "Part des emplois à temps plein"
]
sns.set_context("notebook", font_scale=0.6)
sns.pairplot(data_cleaned[numeric_cols2])
plt.show()

print("REGRESSION LINEAIRE")

X = data_cleaned.drop(columns=['Taux d’insertion', 'Année', "Salaire brut annuel estimé"])
y = data_cleaned['Taux d’insertion']

# Encodage des colonnes catégoriques
X = pd.get_dummies(X, drop_first=True)

# Remplacement des valeurs manquantes
X = X.fillna(0)
y = y.fillna(0)

# Séparation des ensembles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Alignement des colonnes entre X_train et X_test
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Conversion explicite en float
X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)

# Ajout de la constante pour statsmodels
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Ajustement du modèle avec statsmodels
model_sm = sm.OLS(y_train, X_train_sm).fit()

print("Prédictions")
# Prédictions
y_pred = model_sm.predict(X_test_sm)

# Évaluation du modèle
print("R² :", r2_score(y_test, y_pred))
print("Erreur quadratique moyenne :", mean_squared_error(y_test, y_pred))

# Extraire les coefficients et les p-values
coefficients = pd.DataFrame({
    'Variable': ['Intercept'] + list(X_train.columns),
    'Coefficient': model_sm.params.values,
    'p-value': model_sm.pvalues.values
}).sort_values(by='Coefficient', ascending=False)

# Afficher les coefficients avec p-values
coefficients_sans_pval = coefficients.drop(columns=["p-value"])
print(coefficients_sans_pval)
# avec p valeurs
print(coefficients)

# Visualisation des valeurs prédites vs réelles
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.7, label="Prédictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linestyle="--", label="Diagonale parfaite")

# Ajout du coefficient R² sur le graphique
r2 = r2_score(y_test, y_pred)
plt.text(0.05, 0.95, f"$R^2$ = {r2:.2f}", transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Comparaison des valeurs réelles et prédites")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Vérification de la normalité des résidus
residuals = y_test - y_pred
plt.figure(figsize=(12, 8))
sns.kdeplot(residuals, fill=True, color="blue", alpha=0.6, label="Résidus")
plt.axvline(np.mean(residuals), color="red", linestyle="--", label="Moyenne des résidus")
sns.kdeplot(
    np.random.normal(np.mean(residuals), np.std(residuals), len(residuals)),
    color="green", linestyle="--", label="Distribution normale"
)
plt.title("Distribution des Résidus", fontweight="bold")
plt.legend()
plt.grid(True)
plt.show()

# Vérification de l'homoscédasticité
plt.figure(figsize=(12, 8))
plt.scatter(y_pred, residuals, alpha=0.6, color="blue")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Valeurs prédites")
plt.ylabel("Résidus")
plt.title("Valeurs prédites vs Résidus")
plt.grid(True)
plt.show()

# Vérification de l'hétéroscédasticité avec le test de Goldfeld-Quandt
X_test_with_const = sm.add_constant(X_test)
test_stat, p_value, _ = het_goldfeldquandt(residuals, X_test_with_const)
print("Statistique du test Goldfeld-Quandt :", test_stat)
print("p-value :", p_value)
if p_value > 0.05:
    print("Aucune preuve d'hétéroscédasticité (p > 0.05)")
else:
    print("Présence d'hétéroscédasticité (p <= 0.05)")

# Vérification de l'autocorrélation des résidus
sm.graphics.tsa.plot_acf(residuals, lags=40)
plt.title("Autocorrélation des Résidus")
plt.xlabel("Lags")
plt.ylabel("Autocorrélation")
plt.grid(True)
plt.show()

# Vérification de la multicolinéarité avec la matrice de corrélation
correlation_matrix = X_train.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de Corrélation des Variables Explicatives", fontweight="bold")
plt.show()

# Identification des paires de variables fortement corrélées
threshold = 0.8
high_corr = correlation_matrix[(correlation_matrix > threshold) | (correlation_matrix < -threshold)].stack()
high_corr = high_corr[high_corr.index.get_level_values(0) != high_corr.index.get_level_values(1)].drop_duplicates()
print("Variables fortement corrélées :")
print(high_corr)
