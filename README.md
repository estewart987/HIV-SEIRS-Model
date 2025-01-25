# Modeling HIV with an Open Cohort SEIR-S Framework

## Overview
This repository hosts a computational model developed to simulate the transmission dynamics of HIV in the United States. The model adapts the Susceptible-Exposed-Infectious-Recovered-Susceptible (SEIR-S) framework to capture the complexities of HIV, including long latent periods, chronic infectious states, and the effects of anti-retroviral therapy (ART). Using an open cohort structure, the model accounts for population turnover through births, natural deaths, and HIV-related mortality.

Key features of the model include:
- Integration of compartments for latency, viral suppression, and treatment failure.
- Calibration using historical U.S. data (2006–2021) to reflect trends in new infections, viral suppression, and HIV-related deaths.
- Ten-year projections to evaluate the impact of current conditions and interventions.

## Data
The model is informed by:
- U.S. population data from 2006–2021.
- National-level statistics on HIV prevalence, new infections, and AIDS-related deaths.
- Viral suppression rates derived from clinical studies and CDC reports.

Data preprocessing scripts ensure consistency and compatibility with the model, with flexibility to input custom datasets if structured appropriately.

## Model Description
The SEIR-S model extends traditional epidemiological frameworks to incorporate unique aspects of HIV transmission:
- **Susceptible (S)**: Individuals at risk of contracting HIV.
- **Exposed (E)**: Individuals who have contracted HIV but are not yet infectious.
- **Infectious (I)**: Individuals capable of transmitting HIV.
- **Recovered (R)**: Individuals with suppressed viral loads due to ART.
- **Susceptible Again (S)**: Represents re-entry into infectious states due to treatment failure or nonadherence.

The model's system of differential equations governs transitions between these compartments:

$$
\begin{aligned}
\frac{dS}{dt} & = b - \beta \frac{S I}{N} - \mu S \\
\frac{dE}{dt} & = \beta \frac{S I}{N} - \sigma E - \mu E \\
\frac{dI}{dt} & = \sigma E - \nu I - \mu I - \delta I + \gamma R \\
\frac{dR}{dt} & = \nu I - \gamma R - \mu R \\
\frac{dD}{dt} & = \delta I
\end{aligned}
$$

Where:
- $b$: Birth rate.
- $\beta$: Infection rate.
- $\sigma$: Progression rate from exposed to infectious.
- $\nu$: Recovery rate due to ART.
- $\mu$: Natural death rate.
- $\delta$: HIV-related death rate.
- $\gamma$: Loss of viral suppression.
- $N$: Total population.

## Setup and Usage
To use this model:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/estewart987/HIV-SEIRS-Model.git
   cd HIV-SEIRS-Model
   ```

2. **Install Dependencies:**
   ```bash
   pip setup.py install
   ```

3. **Preprocess Data:**
   Prepare your dataset by running:
   ```bash
   python clean_data.py
   ```

4. **Calibrate and Simulate:**
   Use the example script to calibrate the model and generate projections:
   ```bash
   python example.py
   ```

### Example Code
```python
from HIVModel import HIVOpen
import pandas as pd

# Example dataset
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

# Column mapping
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

# Initialize the model
model = HIVOpen(
    beta=0.0001,   # Infection rate
    sigma=0.01,    # Progression rate from exposed to infectious
    nu=0.1,        # Recovery rate (ART)
    mu=0.02,       # Natural death rate
    delta=0.03,    # HIV-related death rate
    gamma=0.01     # Loss of viral suppression
)

# Load data
model.load_data(data, column_mapping)

# Calibrate parameters
model.calibrate_parameters(num_bootstrap=50)

# Simulate future trends
future_predictions = model.simulate(5)
model.simulate_with_uncertainty(5)
```

## License
This project is licensed under the MIT License.
