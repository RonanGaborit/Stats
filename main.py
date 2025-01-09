from ressources.reader_csv import *
from fonction import *


new_data = visualisation_donnees_manquantes(load_data())
# print(new_data.shape)
print(new_data.columns)

colors = setup_plot()
# data = sns.load_dataset(name="mpg")

quantitative_data = new_data.select_dtypes(include=np.number)
qualitative_data = new_data.select_dtypes(include="object")

# global analysis of the data
GlobalAnalysis.print_info(data=new_data)
GlobalAnalysis.print_nan_statistics(data=new_data)
# only 6 missing values in the horsepower column
# we can drop

# analysis of the quantitative variables
QuantitativeAnalysis.print_describe(data=quantitative_data)

QuantitativeAnalysis.plot_linear_correlation(data=quantitative_data, colors=colors)
# high correlation between the variables and target
# multicolinearity between the variables

QuantitativeAnalysis.plot_pairplot(data=new_data, colors=colors)
# we see scale and (anti)correlation between the variables

# analysis of the qualitative variables
QualitativeAnalysis.print_modalities_number(data=qualitative_data)
# name can be preprocessed more
# however, since we have highly correlated variables, "name" can be dropped without losing to much informations

QualitativeAnalysis.plot_modalities_effect_on_target(
    data=new_data, target_column="Taux d’insertion", qualitative_column="Salaire net mensuel médian national", colors=colors
)
# we see that the origin seems to have an effect on the target variable
# we can either encode it or ensure this effect through a central tendency statistical test (Student, Mann-Whitney, ...)
