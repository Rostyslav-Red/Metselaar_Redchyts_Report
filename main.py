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

# model = sm.formula.ols(formula=formula, data=houseData).fit()
# print(f"Regular R^2: {model.rsquared}")
# print(f"Adjusted R^2: {model.rsquared_adj}")
# print(f"Calculated Adj. R^2: {1 - (1 - model.rsquared) * (21613 - 1) / (21613 - 15 - 1)}")


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

    # print("MODE: ", ("Backward", "Forward")[int(forward)], ("Adj. R^2", "p-value")[use_p])

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
        # print("Remaining_columns", remaining_columns)
        # print("Used_columns", used_columns, end="\n\n")
        results = dict()

        for column in remaining_columns:
            # print(f"\tColumn: {column}")
            temp_remaining_columns = remaining_columns.copy()
            temp_remaining_columns.remove(column)

            temp_formula = construct_formula(remaining_columns.copy(), used_columns, column)
            # print(f"\tTemp_formula: {temp_formula}")
            temp_assessment = evaluate_change(temp_formula, column)
            # print(f"\tTemp_assessment: {temp_assessment}")
            results[column] = temp_assessment

        # print(f"Results: {results}")
        min_fit = (min(results, key=results.get), results[min(results, key=results.get)])

        max_fit = (max(results, key=results.get), results[max(results, key=results.get)])

        # print(f"Min: {min_fit}, Max: {max_fit}")

        if forward and use_p:
            if min_fit[1] < 0.05:
                used_columns.append(min_fit[0])
                remaining_columns.remove(min_fit[0])
            else:
                # print("optimal")
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
                # print(f"\tPrev_max: {prev_best_assessment}")
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

columns = ['bathrooms', 'bedrooms', 'sqft_living', 'waterfront', 'sqft_basement',
                'sqft_above', 'floors', 'view', 'condition', 'yr_built', 'yr_renovated', 'grade', 'sqft_living15']

used_columns = stepwise_regression(houseData, columns, 'price', True, True)

formula = 'price ~ ' + ' + '.join(used_columns)

model = sm.formula.ols(formula=formula, data=houseData).fit()

print(formula)
"""
price ~ bathrooms + sqft_living + waterfront + yr_built + grade + view + bedrooms + floors + condition + sqft_living15 + sqft_above + sqft_basement + yr_renovated
"""
# temp = (model.params[i] * houseData[val] for i, val in enumerate(used_columns))
# predicted_score_values = model.params[0] + temp


predicted_score_values = model.params.iloc[0]
for i, val in enumerate(used_columns):
    predicted_score_values += model.params.iloc[i + 1] * houseData[val]
print(predicted_score_values)
regular_residuals = houseData['price']-predicted_score_values
absolute_residuals = np.abs(regular_residuals)

# plt.subplot


def check_assumptions(rows, cols, variables):
    used_columns = stepwise_regression(houseData, variables, 'price', True, True)
    formula = 'price ~ ' + ' + '.join(used_columns)

    model = sm.formula.ols(formula=formula, data=houseData).fit()

    print(formula)
    """
    price ~ bathrooms + sqft_living + waterfront + yr_built + grade + view + bedrooms + floors + condition + sqft_living15 + sqft_above + sqft_basement + yr_renovated
    """
    # temp = (model.params[i] * houseData[val] for i, val in enumerate(used_columns))
    # predicted_score_values = model.params[0] + temp

    predicted_score_values = model.params.iloc[0]
    for i, val in enumerate(used_columns):
        predicted_score_values += model.params.iloc[i + 1] * houseData[val]
    print(predicted_score_values)
    regular_residuals = houseData['price'] - predicted_score_values
    absolute_residuals = np.abs(regular_residuals)

    fig, axes = plt.subplots(rows, cols, figsize = (18, 20))

    # Checking if variability of the residuals is nearly constant
    #(scatterplot of the (absolute) residuals (y-axis) against the predicted values (x-axis))

    # plt.figure(figsize=(8, 6))
    axes[0, 0].scatter(predicted_score_values, absolute_residuals)
    axes[0, 0].set_title('Absolute residuals vs. Predicted values')
    axes[0, 0].set_xlabel('Predicted values')

    axes[0, 0].set_ylabel('Absolute Residuals')
    # plt.show()

    # normality of the residuals (a histogram and QQ-plot of the residuals)

    axes[0, 1].hist(regular_residuals, bins=100, edgecolor='k')
    axes[0, 1].set_title("Histogram of Residuals")
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')

    sm.qqplot(regular_residuals, line='s', ax=axes[1, 0])
    axes[1, 0].set_title('QQ plot Residuals')
    # print(max(regular_residuals), min(regular_residuals))

    # Residuals are independent (scatterplot of the residuals (y-axis) against the order of collection (x-axis))

    axes[1, 1].scatter(np.arange(len(regular_residuals)), regular_residuals)
    axes[1, 1].set_title('Residuals at the order of collection')
    axes[1, 1].set_xlabel('Order of collection')
    axes[1, 1].set_ylabel('Residuals')

    row, col = 2, 0
    temp = used_columns.copy()
    temp.remove('waterfront')
    for var in temp:
        if col == 2:
            row += 1
            col = 0
        axes[row, col].scatter(houseData[var], regular_residuals)
        axes[row, col].set_title(f'Residuals vs {var}')
        axes[row, col].set_xlabel(var)
        axes[row, col].set_ylabel('Residuals')
        col += 1

    sns.boxplot(x='waterfront', y=regular_residuals, data=houseData, ax=axes[8, 0])
    axes[8, 0].set_title('Boxplot of Residuals by Waterfront')
    axes[8, 0].set_xlabel('Waterfront')
    axes[8, 0].set_ylabel('Residuals')
    """
    [2, 0], [2, 1], [3, 0], ... , [7, 1]"""

    plt.tight_layout()
    plt.show()


check_assumptions(9, 2, columns)
