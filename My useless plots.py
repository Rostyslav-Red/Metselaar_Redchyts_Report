import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load data
raw_houseData = pd.read_csv('Rostyslav_Redchyts-Niels_Metselaar-data.csv')
houseData = raw_houseData.drop(columns=['id', 'date', 'zipcode', 'lat', 'long'])

# Adjusting variable categories
continuous_vars = ['sqft_living', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'bedrooms', 'bathrooms']
categorical_vars = ['floors', 'view', 'condition', 'grade']
binary_vars = ['waterfront']

# Prepare the figure layout with controlled subplot sizing
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 35), gridspec_kw={'width_ratios': [1, 1]})
fig.subplots_adjust(hspace=0.4, wspace=0.4)
axes = axes.flatten()  # Flatten to facilitate 1D indexing

# Plotting continuous variables against price with gradient colors
for i, var in enumerate(continuous_vars):
    sns.scatterplot(data=houseData, x=var, y='price', ax=axes[i], alpha=0.5, edgecolor=None,
                    hue='price', palette='coolwarm', legend=None)
    sns.regplot(data=houseData, x=var, y='price', ax=axes[i], scatter=False, color='black')
    axes[i].set_title(f'Price vs. {var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Price')

# Plotting categorical variables with average price and a trend line
for i, var in enumerate(categorical_vars):
    ax = axes[i + len(continuous_vars)]
    grouped_data = houseData.groupby(var)['price'].mean().sort_index()
    sns.barplot(x=grouped_data.index, y=grouped_data.values, hue=grouped_data.index, ax=ax, palette="coolwarm", legend=False)
    sns.lineplot(x=np.arange(len(grouped_data)), y=grouped_data.values, ax=ax, color="black", marker="o")
    ax.set_title(f'Average Price by {var}')
    ax.set_xlabel(var)
    ax.set_ylabel('Average Price')

# Plotting binary variables
for i, var in enumerate(binary_vars):
    ax = axes[i + len(continuous_vars) + len(categorical_vars)]
    sns.boxplot(data=houseData, x=var, y='price', hue=var, ax=ax, palette="coolwarm")
    ax.set_title(f'Price Distribution by {var}')
    ax.set_xlabel(var)
    ax.set_ylabel('Price')

plt.show()
