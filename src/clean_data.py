import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

def load_and_preprocess_population_data(filepath):
    """
    Load population data from a CSV file and preprocess it.
    
    Parameters:
        filepath (str): Path to the population data CSV file.

    Returns:
        pd.DataFrame: Preprocessed population data with calculated columns for births and deaths.
    """
    pop_df = pd.read_csv(filepath)
    pop_df = pop_df.set_index('Unnamed: 0').T.reset_index()
    pop_df = pop_df.rename(columns={'index': 'Year'})
    # Calculate the number of births and deaths
    pop_df['Number of Births'] = pop_df['Number of women in US'] * (pop_df['Crude birth rate per 1,000 women'] / 1000)
    pop_df['Number of Deaths'] = pop_df['US Population'] * (pop_df['Population level Death Rates per 100,000 people'] / 100000)
    return pop_df

def load_and_preprocess_viral_data(filepath):
    """
    Load viral suppression data from a CSV file and preprocess it.
    
    Parameters:
        filepath (str): Path to the viral suppression data CSV file.

    Returns:
        pd.DataFrame: Preprocessed viral suppression data with year converted to integers.
    """
    viral_df = pd.read_csv(filepath)
    viral_df = viral_df.set_index('Unnamed: 0').T.reset_index()
    viral_df = viral_df.rename(columns={'index': 'Year'})
    viral_df['Year'] = viral_df['Year'].astype(int)
    return viral_df

def fit_logistic_function(x_data, y_data):
    """
    Fit a logistic function to the given data points.

    Parameters:
        x_data (array-like): Years corresponding to the data points.
        y_data (array-like): Viral suppression proportions for the given years.

    Returns:
        tuple: The logistic function and the fitted parameters (L, x0, k).
    """
    def logistic_function(x, L, x0, k):
        """
        Logistic function:
        L: Maximum value (asymptote)
        x0: Midpoint (where y = L/2)
        k: Steepness of the curve
        """
        x = x - 2011
        return L / (1 + np.exp(-k * (x - x0)))

    params, _ = curve_fit(logistic_function, x_data, y_data)
    return logistic_function, params

def predict_viral_suppression(viral_df, years_to_predict):
    """
    Predict viral suppression values for specific years using linear regression.

    Parameters:
        viral_df (pd.DataFrame): DataFrame with existing viral suppression data.
        years_to_predict (list): Years to predict viral suppression values for.

    Returns:
        pd.DataFrame: Updated DataFrame with new predicted rows added and sorted by year.
    """
    # Filter data for years 1997-1999 to train the model
    filtered_df = viral_df[viral_df['Year'].isin([1997, 1998, 1999])]
    x = filtered_df['Year'].astype(int).values.reshape(-1, 1)
    y = filtered_df['Viral Suppression Proportion'].values

    model = LinearRegression()
    model.fit(x, y)
    # Predict viral suppression values for specified years
    predicted_values = model.predict(np.array(years_to_predict).reshape(-1, 1))

    # Add the predictions as new rows and sort the DataFrame by year
    new_rows = pd.DataFrame({"Year": years_to_predict, "Viral Suppression Proportion": predicted_values})
    return pd.concat([viral_df, new_rows], ignore_index=True).sort_values(by="Year", ignore_index=True)

def adjust_viral_suppression(viral_df, cdc_data, logistic_function, params):
    """
    Adjust viral suppression values based on historical and predicted data.

    Parameters:
        viral_df (pd.DataFrame): DataFrame with viral suppression data.
        cdc_data (dict): CDC viral suppression proportions for specific years.
        logistic_function (function): Fitted logistic function for estimation.
        params (tuple): Parameters of the fitted logistic function (L, x0, k).

    Returns:
        pd.DataFrame: DataFrame with adjusted viral suppression values for all years.
    """
    L, x0, k = params
    adj_viral_df = pd.DataFrame({'Year': list(range(1990, 2022)), 'Viral Suppression Proportion': np.nan})

    # Calculate the ratio of CDC viral suppression proportion in 2011 to clinical study suppression in 2011
    viral_sup_ratio = (cdc_data[2011] / viral_df.loc[viral_df['Year'] == 2011, 'Viral Suppression Proportion']).iloc[0]

    early_90s_addition = 0.0025
    early_90s_multiplier = 0

    # Adjust values based on year
    for year in adj_viral_df['Year']:
        if year < 1995:
            adj_viral_df.loc[adj_viral_df['Year'] == year, 'Viral Suppression Proportion'] = 0.1 * 0.1 + early_90s_addition * early_90s_multiplier
            early_90s_multiplier += 1
        elif 1995 <= year < 2011:
            # Adjust clinical trial suppression proportions based on ratio with CDC data
            adj_viral_sup_value = viral_sup_ratio * viral_df.loc[viral_df['Year'] == year, 'Viral Suppression Proportion']
            adj_viral_df.loc[adj_viral_df['Year'] == year, 'Viral Suppression Proportion'] = adj_viral_sup_value.iloc[0]
        elif year > 2010 and year not in [2011, 2016, 2019]:
            # Estimate using the logistic function for 2012-2021
            adj_viral_df.loc[adj_viral_df['Year'] == year, 'Viral Suppression Proportion'] = logistic_function(x=year, L=L, x0=x0, k=k)
        elif year in [2011, 2016, 2019]:
            # Use CDC data for specific years
            adj_viral_df.loc[adj_viral_df['Year'] == year, 'Viral Suppression Proportion'] = cdc_data[year]

    return adj_viral_df

def main():
    """
    Main workflow to preprocess data, fit models, and adjust viral suppression values.
    """
    pop_filepath = 'data/population_data.csv'
    viral_filepath = 'data/viral_suppression.csv'

    # Load data
    pop_df = load_and_preprocess_population_data(pop_filepath)
    viral_df = load_and_preprocess_viral_data(viral_filepath)

    # CDC data
    x_data = np.array([2011, 2016, 2019])
    y_data = np.array([0.3, 0.53, 0.66])
    cdc_data = {year: val for year, val in zip(x_data, y_data)}

    # Fit logistic function
    logistic_function, params = fit_logistic_function(x_data, y_data)

    # Predict and append 1995-1996 data
    viral_df = predict_viral_suppression(viral_df, [1995, 1996])

    # Adjust viral suppression values
    adj_viral_df = adjust_viral_suppression(viral_df, cdc_data, logistic_function, params)

    # Add adjusted viral suppression values to population DataFrame
    pop_df['Viral Suppression Proportion'] = adj_viral_df['Viral Suppression Proportion']

    # Print or save the final DataFrame
    pop_df.to_csv('data/updated_population_data.csv', index=False)
    
if __name__ == "__main__":
    main()
