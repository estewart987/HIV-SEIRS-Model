from HIVOpen.HIVOpen import HIVOpen
import pandas as pd

# Initialize the model with initial guesses for parameters
model = HIVOpen(beta=0.1, sigma=0.2, nu=0.1, mu=0.01, delta=0.05, gamma=0.02)

# Define the column mapping for your dataset
column_dict = {
    "year": "Year",
    "population": "US Population",
    "number_of_births": "Number of Births",
    "natural_death_rate": "Population level Death Rates per 100,000 people",
    "new_infections": "New HIV Infections",
    "viral_suppression": "Viral Suppression Proportion"
}

# Load the historical data
model.load_data("HIVOpen/Data/example_data.csv", column_dict)

# Calibrate the model parameters using bootstrapping
model.calibrate_parameters(num_bootstrap=100)

# Simulate with uncertainty for 10 years
model.simulate_with_uncertainty(10)


