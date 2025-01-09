from ressources.reader_csv import *
from fonction import *


print(verifcation())
dataframe = load_data()
print(dataframe.head())  # Display the first few rows of the DataFrame
print(dataframe.columns)
print(dataframe.shape)
print(dataframe.info())

new_data = visualisation_donnees_manquantes(dataframe)
print(new_data.shape)
print(new_data.columns)

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

