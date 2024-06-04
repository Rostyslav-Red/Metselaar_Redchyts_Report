import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import probplot

sns.set_style("whitegrid")

houseData = pd.read_csv('Rostyslav_Redchyts-Niels_Metselaar-data.csv')
print(houseData.tail())

"""
Explanatory Variables:
1.	Number of bathrooms (bathrooms)
2.	Number of bedrooms (bedrooms)
3.	Square footage of living spaces (sqft_living)
4.	Square footage of the lot (sqft_lot)
5.	Square footage of the basement (sqft_basement)
6.	Square footage above ground level (sqft_above)
7.	Number of floors (floors)
8.	View quality (scale of 0-4) (view)
9.	Condition of the house (scale of 1-5) (condition)
10.	Year built (yr_built)
11.	Year renovated (yr_renovated)
12.	Grade of building’s construction and design (scale of 1-13) (grade)
"""

column_names = ['bathrooms', 'bedrooms', 'sqft_living', 'sqft_lot', 'sqft_basement', 'sqft_above', 'floors', 'view', 'condition',
                'yr_built', 'yr_renovated', 'grade']

print(len(column_names))

for i in column_names:
    sns.scatterplot(data=houseData, x = i, y = 'price')
    plt.show()