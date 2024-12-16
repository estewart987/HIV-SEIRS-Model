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
pip OR pip3 setup.py insteall
```

### 5. Run example.py
Run example.py. This will clean the data to be in a format appropriate to calibrate the model and run simulations. You can adjust the initial guesses for the parameters in example.py and load in your own dataset, as long as the columns match up with the code in HIVModel.py
```bash
python OR python3 example.py
```

### 6. Example
```python

```
