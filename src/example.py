from HIVModel import HIVOpen
import pandas as pd

# Initialize the model with initial guesses for parameters
model = HIVOpen(beta=0.01, sigma=0.2, nu=0.1, mu=0.01, delta=0.05, gamma=0.02)

# Define the column mapping for your dataset
column_dict = {
    "year": "Year",
    "population": "US Population",
    "number_of_births": "Number of Births",
    "natural_death_rate": "Population level Death Rates per 100,000 people",
    "new_infections": "New HIV Infections",
    "viral_suppression": "Viral Suppression Proportion",
    "deaths_hiv": "Deaths from HIV/AIDS",
    "number_of_deaths": "Number of Deaths",
    "total_hiv": "People living with HIV",
}

# Load the historical data
model.load_data("data/example_data_test.csv", column_dict)

# Calibrate the model parameters using bootstrapping
model.calibrate_parameters(num_bootstrap=200)

# Simulate with uncertainty for 10 years
model.simulate_with_uncertainty(10)
