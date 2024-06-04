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
2.	Number of bedrooms (bedrooms) # Exclude the weird one
 (transform the bathroom) variable using quadratic transformation
3.	Square footage of living spaces (sqft_living)
4.	Square footage of the lot (sqft_lot) # Delete it
5.	Square footage of the basement (sqft_basement) # Divide it into 
 0 - doesn't have a basement and 1 - has a basement
6.	Square footage above ground level (sqft_above) 
7.	Number of floors (floors) # Quadratic transformation
8.	View quality (scale of 0-4) (view)
9.	Condition of the house (scale of 1-5) (condition)
10.	Year built (yr_built) # Exclude outliers
11.	Year renovated (yr_renovated) # Transform into the number of years since renovation
12.	Grade of building’s construction and design (scale of 1-13) (grade)
"""

column_names = ['bathrooms', 'bedrooms', 'I(bedrooms**2)', 'sqft_living', 'sqft_basement_dummy',
           'sqft_above', 'floors', 'I(floors**2)', 'view', 'condition', 'yr_built', 'yr_renovated_dummy', 'grade']

print(len(column_names))

formula = ('price ~ bathrooms + bedrooms + I(bedrooms**2) + sqft_living + sqft_basement_dummy + '
           'sqft_above + floors + I(floors**2) + view + condition + yr_built + '
           'yr_renovated_dummy + grade')

houseData['sqft_basement_dummy'] = (houseData['sqft_basement'] > 0).astype(int)
current_year = 2024
houseData['yr_renovated_dummy'] = np.where(houseData['yr_renovated'] == 0,
                                           current_year - houseData['yr_built'],
                                           current_year - houseData['yr_renovated'])
# for i in column_names:
#     sns.scatterplot(data=houseData, x = i, y = 'price')
#     plt.show()


#checking normality of response variable
sm.qqplot(houseData['price'], line='s')
plt.title('QQ plot price distribution')
plt.show()

sns.displot(houseData['price'], bins=50)
plt.title('The distribution of housing prices')
plt.show()


model = sm.formula.ols(formula = formula, data = houseData)
model_fit = model.fit()

# optimal_flag = False
#
# useful_columns = column_names.copy()
# prev_adj_r2 = model_fit.rsquared_adj
#
# while not optimal_flag:
#     results = dict()
#
#     for i in useful_columns:
#         shortened_columns = useful_columns.copy()
#         shortened_columns.remove(i)
#         new_formula = "price ~ " + " + ".join(shortened_columns)
#         new_adj_r2 = sm.formula.ols(formula=new_formula, data=houseData).fit().rsquared_adj
#         results[i] = new_adj_r2
#
#     best_fit = (max(results, key=results.get), results[max(results, key=results.get)])
#     print(results)
#     print(best_fit)
#     if best_fit[1] >= prev_adj_r2:
#         useful_columns.remove(best_fit[0])
#         prev_adj_r2 = best_fit[1]
#     else:
#         optimal_flag = True
#
# print(useful_columns, len(useful_columns))

optimal_flag = False

useful_columns = []
remaining_columns = column_names.copy()
prev_adj_r2 = 0

while not optimal_flag and remaining_columns:
    results = dict()

    for i in remaining_columns:
        new_formula = "price ~ " + " + ".join(useful_columns + [i])
        new_adj_r2 = sm.formula.ols(formula=new_formula, data=houseData).fit().rsquared_adj
        results[i] = new_adj_r2

    best_fit = (max(results, key=results.get), results[max(results, key=results.get)])
    print(results)
    print(best_fit)
    if best_fit[1] >= prev_adj_r2:
        remaining_columns.remove(best_fit[0])
        useful_columns.append(best_fit[0])
        prev_adj_r2 = best_fit[1]
    else:
        optimal_flag = True

print(useful_columns, len(useful_columns))

print(model_fit.summary())
print(model_fit.rsquared_adj, model_fit.rsquared)