import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


class PopulationDataPreprocessor:
    """A class to handle the preprocessing of population and viral suppression data.

    This class provides functionality to load, process, and analyze population and
    viral suppression data from CSV files. It includes methods for data cleaning,
    prediction of viral suppression rates, and adjustment of historical data.

    Attributes:
        population_filepath (str): Path to the population data CSV file
        viral_filepath (str): Path to the viral suppression data CSV file
        pop_df (pandas.DataFrame): DataFrame containing population data
        viral_df (pandas.DataFrame): DataFrame containing viral suppression data
        logistic_function (callable): Fitted logistic function for predictions
        params (tuple): Parameters of the fitted logistic function
        cdc_years (numpy.ndarray): Array of years with CDC viral suppression data
        cdc_values (numpy.ndarray): Array of CDC viral suppression values
        cdc_data (dict): Dictionary mapping years to CDC viral suppression values
    """

    def __init__(self, population_filepath, viral_filepath):
        """Initialize the preprocessor with file paths.

        Args:
            population_filepath (str): Path to the population data CSV file
            viral_filepath (str): Path to the viral suppression data CSV file
        """
        self.population_filepath = population_filepath
        self.viral_filepath = viral_filepath
        self.pop_df = None
        self.viral_df = None
        self.logistic_function = None
        self.params = None

        # CDC viral suppression data
        self.cdc_years = np.array([2011, 2016, 2019])
        self.cdc_values = np.array([0.3, 0.53, 0.66])
        self.cdc_data = dict(zip(self.cdc_years, self.cdc_values))

    def load_population_data(self):
        """Load and preprocess population data from CSV file.

        This method reads the population data CSV file, transposes it for proper
        formatting, and calculates additional metrics including births and deaths.

        Returns:
            pandas.DataFrame: Processed population data with calculated metrics
        """
        self.pop_df = pd.read_csv(self.population_filepath)
        self.pop_df = self.pop_df.set_index("Unnamed: 0").T.reset_index()
        self.pop_df = self.pop_df.rename(columns={"index": "Year"})

        # Calculate births and deaths
        self.pop_df["Number of Births"] = self.pop_df["Number of women in US"] * (
            self.pop_df["Crude birth rate per 1,000 women"] / 1000
        )
        self.pop_df["Number of Deaths"] = self.pop_df["US Population"] * (
            self.pop_df["Population level Death Rates per 100,000 people"] / 100000
        )
        return self.pop_df

    def load_viral_data(self):
        """Load and preprocess viral suppression data from CSV file.

        This method reads the viral suppression data CSV file, transposes it for proper
        formatting, and converts the year column to integer type.

        Returns:
            pandas.DataFrame: Processed viral suppression data
        """
        self.viral_df = pd.read_csv(self.viral_filepath)
        self.viral_df = self.viral_df.set_index("Unnamed: 0").T.reset_index()
        self.viral_df = self.viral_df.rename(columns={"index": "Year"})
        self.viral_df["Year"] = self.viral_df["Year"].astype(int)
        return self.viral_df

    @staticmethod
    def _logistic_function(x, L, x0, k):
        """Calculate the logistic function value.

        Args:
            x (float or numpy.ndarray): Input value (year)
            L (float): Maximum value (asymptote)
            x0 (float): Midpoint (where y = L/2)
            k (float): Steepness of the curve

        Returns:
            float or numpy.ndarray: Calculated logistic function value(s)
        """
        x = x - 2011
        return L / (1 + np.exp(-k * (x - x0)))

    def fit_logistic_curve(self):
        """Fit logistic function to CDC data points.

        This method fits a logistic function to the CDC viral suppression data points
        using scipy's curve_fit function.

        Returns:
            tuple: A tuple containing:
                - callable: The fitted logistic function
                - tuple: The optimal parameters (L, x0, k)
        """
        self.params, _ = curve_fit(
            self._logistic_function, self.cdc_years, self.cdc_values
        )
        return self._logistic_function, self.params

    def predict_early_viral_suppression(self, years_to_predict):
        """Predict viral suppression values for early years using linear regression.

        Args:
            years_to_predict (list): Years to predict viral suppression values for

        Returns:
            pandas.DataFrame: Updated viral suppression DataFrame with predicted values
                            for early years
        """
        filtered_df = self.viral_df[self.viral_df["Year"].isin([1997, 1998, 1999])]
        x = filtered_df["Year"].astype(int).values.reshape(-1, 1)
        y = filtered_df["Viral Suppression Proportion"].values

        model = LinearRegression()
        model.fit(x, y)
        predicted_values = model.predict(np.array(years_to_predict).reshape(-1, 1))

        new_rows = pd.DataFrame(
            {"Year": years_to_predict, "Viral Suppression Proportion": predicted_values}
        )
        self.viral_df = pd.concat([self.viral_df, new_rows], ignore_index=True)
        self.viral_df = self.viral_df.sort_values(by="Year", ignore_index=True)
        return self.viral_df

    def adjust_viral_suppression(self):
        """Adjust viral suppression values based on historical and predicted data.

        This method creates adjusted viral suppression values using a combination of
        historical data, CDC data points, and logistic function predictions.

        Returns:
            pandas.DataFrame: DataFrame containing adjusted viral suppression values
                            for years 1990-2021
        """
        L, x0, k = self.params
        adj_viral_df = pd.DataFrame(
            {"Year": list(range(1990, 2022)), "Viral Suppression Proportion": np.nan}
        )

        viral_sup_ratio = (
            self.cdc_data[2011]
            / self.viral_df.loc[
                self.viral_df["Year"] == 2011, "Viral Suppression Proportion"
            ]
        ).iloc[0]

        early_90s_addition = 0.0025
        early_90s_multiplier = 0

        for year in adj_viral_df["Year"]:
            if year < 1995:
                adj_viral_df.loc[
                    adj_viral_df["Year"] == year, "Viral Suppression Proportion"
                ] = 0.1 * 0.1 + early_90s_addition * early_90s_multiplier
                early_90s_multiplier += 1
            elif 1995 <= year < 2011:
                adj_viral_sup_value = (
                    viral_sup_ratio
                    * self.viral_df.loc[
                        self.viral_df["Year"] == year, "Viral Suppression Proportion"
                    ]
                )
                adj_viral_df.loc[
                    adj_viral_df["Year"] == year, "Viral Suppression Proportion"
                ] = adj_viral_sup_value.iloc[0]
            elif year > 2010 and year not in [2011, 2016, 2019]:
                adj_viral_df.loc[
                    adj_viral_df["Year"] == year, "Viral Suppression Proportion"
                ] = self._logistic_function(year, L, x0, k)
            elif year in [2011, 2016, 2019]:
                adj_viral_df.loc[
                    adj_viral_df["Year"] == year, "Viral Suppression Proportion"
                ] = self.cdc_data[year]

        return adj_viral_df

    def process_data(self):
        """Execute the main data processing workflow.

        This method orchestrates the complete data processing workflow, including:
        1. Loading population and viral suppression data
        2. Fitting the logistic curve
        3. Predicting early years
        4. Adjusting viral suppression values
        5. Combining all processed data

        Returns:
            pandas.DataFrame: Final processed DataFrame containing population data
                            with adjusted viral suppression values
        """
        # Load both datasets
        self.load_population_data()
        self.load_viral_data()

        # Fit logistic function and get parameters
        self.logistic_function, self.params = self.fit_logistic_curve()

        # Predict early years and adjust viral suppression
        self.predict_early_viral_suppression([1995, 1996])
        adj_viral_df = self.adjust_viral_suppression()

        # Add adjusted viral suppression to population data
        self.pop_df["Viral Suppression Proportion"] = adj_viral_df[
            "Viral Suppression Proportion"
        ]
        return self.pop_df


def main():
    """Run the preprocessing workflow.

    This function instantiates the PopulationDataPreprocessor class with the
    appropriate file paths, processes the data, and saves the results to a CSV file.
    """
    preprocessor = PopulationDataPreprocessor(
        population_filepath="data/population_data.csv",
        viral_filepath="data/viral_suppression.csv",
    )

    # Process all data
    final_df = preprocessor.process_data()

    # Save the results
    final_df.to_csv("data/updated_population_data.csv", index=False)


if __name__ == "__main__":
    main()
