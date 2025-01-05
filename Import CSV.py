import pandas
from pandas import DataFrame
def load_data(path: str = "C:\\Users\\ronan\\Desktop\\Projet Stats\\fr-esr-insertion_professionnelle-master_donnees_nationales (2).csv")-> DataFrame:
 """
 Load data from a CSV file with pandas.
 Parameters----------
 path : str
 The path to the CSV file.
 Returns------
DataFrame
 The data loaded from the CSV file.
 """
 return pandas.read_csv(path, delimiter=";")
# Call the function
path = "C:\\Users\\ronan\\Desktop\\Projet Stats\\fr-esr-insertion_professionnelle-master_donnees_nationales (2).csv"
data = load_data(path)

# Print the DataFrame and file path
print(f"Data loaded from: {path}")
print(data.head())  # Display the first few rows of the DataFrame
print(data.columns)
print(data.shape)
