# Team 9 Final Project: Modeling HIV with an Open Cohort SEIR-S Framework
Team Members: Ryan O'Dea, Emily Stewart, Patrick Thornton

## Table of Contents

1. [Overview](#overview)
2. [Setup and Usage](#setup-and-usage)

## Overview
This project adapts the SEIR-S framework to model HIV dynamics in the United States, incorporating key features such as long latent periods, chronic infectious states, and the impact of anti-retroviral therapy (ART). The model uses an open cohort structure to account for population turnover through births, natural deaths, and HIV-related mortality.

## Setup and Usage
Follow these steps to set up the project and start using it:

### 1. Clone the Repository
Download the project files to your local machine:
```bash
git clone https://code.harvard.edu/AM215/final_9.git
```

### 2. Navigate to the Project Directory
Move into the project folder:
```bash
cd src
```

### 3. Install Required Packages
Use the `setup.py` file to install all dependencies:
```bash
pip OR pip3 setup.py install
```

### 4. Run clean_data.py
Run clean_data.py. This will create a dataset to be in a format appropriate for calibrating the model.
```bash
python OR python3 clean_data.py
```

### 5. Run example.py
Run example.py. This will calibrate the model and run simulations. You can adjust the initial guesses for the parameters in example.py and load in your own dataset, as long as the columns match up with the code in HIVModel.py.
```bash
python OR python3 example.py
```

### 6. Example usage
Below is a bit of code demonstrating typical usage of our model. Feel free to adapt this to your own use case!
```python
import pandas as pd
from HIVModel import HIVOpen

# Take some arbitrary sample data
data = pd.DataFrame({
    'Year': [2010, 2011, 2012],
    'Population': [300000000, 301000000, 302000000],
    'New_HIV_Infections': [50000, 48000, 46000],
    'Total_HIV_Cases': [1000000, 1020000, 1040000],
    'HIV_Deaths': [15000, 14500, 14000],
    'Births': [4000000, 4010000, 4020000],
    'Natural_Death_Rate': [800, 790, 780],
    'Viral_Suppression': [0.50, 0.52, 0.54]
})

# Define whatever column mappings you need for your data
column_mapping = {
    'year': 'Year',
    'population': 'Population',
    'new_infections': 'New_HIV_Infections',
    'total_hiv': 'Total_HIV_Cases',
    'deaths_hiv': 'HIV_Deaths',
    'number_of_births': 'Births',
    'natural_death_rate': 'Natural_Death_Rate',
    'viral_suppression': 'Viral_Suppression'
}

# Initialize model with parameters
model = HIVOpen(
    beta=0.0001,   # Infection rate
    sigma=0.01,    # Rate of progression from exposed to infectious
    nu=0.1,        # Recovery rate (Anti-retroviral Therapies)
    mu=0.02,       # Natural death rate
    delta=0.03,    # HIV-related death rate
    gamma=0.01     # Loss of immunity rate
)

# Load data
model.load_data(data, column_mapping)

# Calibrate model parameters
model.calibrate_parameters(num_bootstrap=50)

# Run simulation for 5 years into the future
future_predictions = model.simulate(5)

# Run simulation for 5 years into the future with uncertainty analysis
model.simulate_with_uncertainty(5)  # this will also generate plots!
```
