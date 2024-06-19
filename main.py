import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import probplot

sns.set_style("whitegrid")

# Retrieving the data from the database
raw_houseData = pd.read_csv('Rostyslav_Redchyts-Niels_Metselaar-data.csv')
houseData = raw_houseData.drop(columns=['id', 'date', 'zipcode', 'lat', 'long'])

#####

sns.scatterplot(data=houseData, x='sqft_basement', y='price', alpha=0.5, edgecolor=None,
                hue='price', palette='coolwarm', legend=None).set_title('Price vs. sqft_basement')
plt.xlabel('sqft_basement')
plt.ylabel('Price')
sns.regplot(data=houseData, x='sqft_basement', y='price', scatter=False, color='black')
plt.show()

#####

houseData.drop(index=houseData[raw_houseData['bedrooms'] == 33].index, inplace=True)
houseData.reset_index(drop=True, inplace=True)

print(houseData.tail())

"""
Explanatory Variables:
1.	Number of bathrooms (bathrooms) 
2.	Number of bedrooms (bedrooms) # Exclude the weird one
    (transform the bathroom) variable using quadratic transformation
3.	Square footage of living spaces (sqft_living)
4.	Square footage of the basement (sqft_basement) # Divide it into 
    0 - doesn't have a basement and 1 - has a basement
5.	Square footage above ground level (sqft_above) 
6.	Number of floors (floors) # Quadratic transformation
7.	View quality (scale of 0-4) (view)
8.	Condition of the house (scale of 1-5) (condition)
9.	Year built (yr_built) # Exclude outliers
10.	Year renovated (yr_renovated) # Transform into the number of years since renovation
11.	Grade of building’s construction and design (scale of 1-13) (grade)
12. The square footage of interior housing living space for the nearest 15 neighbors (sqft_living15)
"""

# The list of the variables that are considered for the model

column_names = ['bathrooms', 'bedrooms', 'I(bedrooms ** 2)', 'sqft_living', 'sqft_basement_dummy', 'sqft_above',
                'floors',
                'I(floors ** 2)', 'view', 'condition', 'yr_built', 'yr_renovated_dummy', 'grade', 'sqft_living15'
                , 'waterfront']

formula = ('log_price ~ bathrooms + bedrooms + I(bedrooms ** 2) + sqft_living + sqft_basement_dummy + '
           'sqft_above + floors + I(floors ** 2) + view + condition + yr_built + '
           'yr_renovated_dummy + grade + sqft_living15 + sqft_lot15 + waterfront')

# Adding the column, named 'sqft_basement_dummy' to the database. This variable equates to 1 if the property has a
# basement and 0 when it doesn't
houseData['sqft_basement_dummy'] = (houseData['sqft_basement'] > 0).astype(int)

# Adding the column, named 'yr_renovated_dummy', which equates to the number of years that have passed since the
# last renovation of the property. If the renovation never happened, the output is just the age of the property.
current_year = 2024
houseData['yr_renovated_dummy'] = np.where(houseData['yr_renovated'] == 0,
                                           current_year - houseData['yr_built'],
                                           current_year - houseData['yr_renovated'])

houseData['log_price'] = np.log(houseData['price'])

model = sm.formula.ols(formula=formula, data=houseData).fit()
print(f"Regular R^2: {model.rsquared}")
print(f"Adjusted R^2: {model.rsquared_adj}")
print(f"Calculated Adj. R^2: {1 - (1 - model.rsquared) * (21613 - 1) / (21613 - 15 - 1)}")


def basic_plot():
    for column in column_names:
        if "**" not in column:
            # Plotting the data
            sns.regplot(data=houseData, x=column, y='price', scatter_kws={"color": "blue"}, line_kws={"color": "red"})
            plt.show()

            # Calculating and printing the Pearson correlation coefficient
            correlation = houseData[[column, 'price']].corr().iloc[0, 1]
            print(f"Correlation between {column} and price: {correlation:.2f}")


# basic_plot()


def response_normality_check():  # Checking the normality of the response variable
    sm.qqplot(houseData['log_price'], line='s')
    plt.title('QQ plot price distribution')
    plt.show()

    sns.displot(houseData['log_price'], bins=50)
    plt.title('The distribution of housing prices')
    plt.show()


# response_normality_check()


def stepwise_regression(data, columns, response, forward=True, use_p=True):
    def construct_formula(rem_c, used_c, current_c=''):
        if forward:
            if used_c:
                return response + " ~ " + " + ".join(used_c) + " + " + current_c
            else:
                return response + " ~ " + current_c
        else:
            if use_p:
                return response + " ~ " + " + ".join(rem_c)
            else:
                rem_c.remove(current_c)
                return response + " ~ " + " + ".join(rem_c)

    def evaluate_change(current_formula, c):
        fitted_model = sm.formula.ols(formula=current_formula, data=data).fit()
        if use_p:
            # print(fitted_model.pvalues)
            return fitted_model.pvalues[c]
        else:
            return fitted_model.rsquared_adj

    print("MODE: ", ("Backward", "Forward")[int(forward)], ("Adj. R^2", "p-value")[use_p])

    optimal_flag = False

    remaining_columns = columns.copy()
    used_columns = list()

    if not forward:
        prev_best_assessment = sm.formula.ols(formula=response + " ~ " + " + ".join(remaining_columns),
                                              data=data).fit().rsquared_adj
    else:
        prev_best_assessment = 0

    # proper_length_flag = (use_p and remaining_columns) or (not use_p and not forward and len(remaining_columns)>1)

    while not optimal_flag and remaining_columns:
        print("Remaining_columns", remaining_columns)
        print("Used_columns", used_columns, end="\n\n")
        results = dict()

        for column in remaining_columns:
            print(f"\tColumn: {column}")
            temp_remaining_columns = remaining_columns.copy()
            temp_remaining_columns.remove(column)

            temp_formula = construct_formula(remaining_columns.copy(), used_columns, column)
            print(f"\tTemp_formula: {temp_formula}")
            temp_assessment = evaluate_change(temp_formula, column)
            print(f"\tTemp_assessment: {temp_assessment}")
            results[column] = temp_assessment

        print(f"Results: {results}")
        min_fit = (min(results, key=results.get), results[min(results, key=results.get)])

        max_fit = (max(results, key=results.get), results[max(results, key=results.get)])

        print(f"Min: {min_fit}, Max: {max_fit}")

        if forward and use_p:
            if min_fit[1] < 0.05:
                used_columns.append(min_fit[0])
                remaining_columns.remove(min_fit[0])
            else:
                print("optimal")
                optimal_flag = True
        elif forward and (not use_p):
            if max_fit[1] >= prev_best_assessment:
                used_columns.append(max_fit[0])
                remaining_columns.remove(max_fit[0])
                prev_best_assessment = max_fit[1]
            else:
                optimal_flag = True
        elif (not forward) and use_p:
            if max_fit[1] > 0.05:
                remaining_columns.remove(max_fit[0])
            else:
                optimal_flag = True
        elif (not forward) and (not use_p):
            if max_fit[1] > prev_best_assessment:
                print(f"\tPrev_max: {prev_best_assessment}")
                remaining_columns.remove(max_fit[0])
                prev_best_assessment = max_fit[1]
            else:
                optimal_flag = True

    if forward:
        return used_columns
    else:
        return remaining_columns


# print(
#     "Forward p-values: \n\t" + str(stepwise_regression(houseData, column_names, "log_price", True, True)) + "\n" +
#     "Forward Adjusted R^2: \n\t" + str(stepwise_regression(houseData, column_names, "log_price", True, False)) + "\n" +
#     "Backward p-values: \n\t" + str(stepwise_regression(houseData, column_names, "log_price", False, True)) + "\n" +
#     "Backward Adjusted R^2: \n\t" + str(stepwise_regression(houseData, column_names, "log_price", False, False))
# )


# print(model_fit.summary())
# print(model_fit.rsquared_adj, model_fit.rsquared)

# test_f = 'log_price ~ bathrooms + sqft_living + yr_built + grade + view + I(floors ** 2) + sqft_living15 + sqft_basement_dummy + waterfront + condition + bedrooms + sqft_above + I(bedrooms ** 2) + floors'
# model_test = sm.formula.ols(formula=test_f, data=houseData).fit()
# print(f"Adjusted R^2: {model.rsquared_adj}")
#
# test_f += ' + yr_renovated_dummy'
# model_test_2 = sm.formula.ols(formula=test_f, data=houseData).fit()
# print(model_test_2.pvalues['yr_renovated_dummy'])