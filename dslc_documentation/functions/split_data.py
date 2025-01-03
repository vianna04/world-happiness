import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(df):
    # Check if the 'year' column exists
    if 'year' not in df.columns:
        raise ValueError("Dataframe must contain a 'year' column")

    # Sort by the 'year' column in ascending order
    df_sorted = df.sort_values(by='year')

    # Obtain all years and split them into three subsets with a ratio of 6:1:3
    unique_years = sorted(df['year'].unique())
    n_years = len(unique_years)
    train_years = unique_years[:int(0.6 * n_years)]
    val_years = unique_years[int(0.6 * n_years):int(0.7 * n_years)]
    test_years = unique_years[int(0.7 * n_years):]

    # Group the data by year
    train = df_sorted[df_sorted['year'].isin(train_years)]
    val = df_sorted[df_sorted['year'].isin(val_years)]
    test = df_sorted[df_sorted['year'].isin(test_years)]

    # Output the three datasets
    return train, val, test
