import pandas as pd
from functions.impute_feature import impute_feature

# Define a function to perform batch imputation and sorting
def clean_happiness(happiness_orig, impute_method):
    # Rename the column names
    happiness_clean = happiness_orig.rename(columns={
        "Life Ladder": "happiness",
        "Log GDP per capita": "log_gdp_per_capita",
        "Social support": "social_support",
        "Healthy life expectancy at birth": "life_expectancy",
        "Freedom to make life choices": "freedom_choices",
        "Generosity": "generosity",
        "Perceptions of corruption": "corruption",
        "Positive affect": "positive_affect",
        "Negative affect": "negative_affect",
        "Confidence in national government": "government_confidence",
        "Democratic Quality": "democratic_quality",
        "Delivery Quality": "delivery_quality",
        "Standard deviation of ladder by country-year": "sd_ladder",
        "Standard deviation/Mean of ladder by country-year": "sd_d_mean_ladder",
        "GINI index (World Bank estimate)": "gini_wb_estimate",
        "GINI index (World Bank estimate), average 2000-15": "gini_wb_estimate_average",
        "gini of household income reported in Gallup, by wp5-year": "gini_hh_income"
    })
    
    # Create a complete combination of countries and years
    country_year_combinations = pd.MultiIndex.from_product(
        [happiness_clean['country'].unique(), happiness_clean['year'].unique()],
        names=["country", "year"]
    )
    country_year_combinations_df = pd.DataFrame(index=country_year_combinations).reset_index()
    happiness_clean = country_year_combinations_df.merge(happiness_clean, on=["country", "year"], how="left")
    
     # Sort by country and year
    happiness_clean = happiness_clean.sort_values(by=["country", "year"]).reset_index(drop=True)
    
    # Obtain the columns that need to be imputed (excluding the country and year columns)
    columns_to_impute = [col for col in happiness_clean.columns if col not in ["country", "year"]]
    
    #  Automate imputation and add new columns
    imputed_columns = []
    for col in columns_to_impute:
        imputed_col_name = f"{col}_imputed"
        happiness_clean[imputed_col_name] = impute_feature(happiness_clean, 
                                                           feature=col, 
                                                           group="country", 
                                                           impute_method=impute_method)
        imputed_columns.append(imputed_col_name)
        
    # Rearrange the order of columns
    column_order = ['country', 'year'] + imputed_columns + columns_to_impute
    happiness_clean = happiness_clean[column_order]
    
    # Remove the original columns and the country column
    happiness_clean = happiness_clean.drop(columns=["country"] + columns_to_impute)
    
    # Remove the parts of the column names that contain _imputed
    happiness_clean.columns = [col.replace("_imputed", "") for col in happiness_clean.columns]
    
    return happiness_clean
