import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class HIVOpen:
    def __init__(self, beta, sigma, nu, mu, delta, gamma):
        """
        Initialize SEIRS parameters.

        Parameters:
        - beta: Infection rate
        - sigma: Rate of progression from exposed to infectious
        - nu: Recovery rate (Anti-retroviral Therapies)
        - mu: Natural death rate
        - delta: HIV-related death rate
        - gamma: Loss of immunity rate
        """
        self.beta = beta        
        self.sigma = sigma      
        self.nu = nu            
        self.mu = mu            
        self.delta = delta      
        self.gamma = gamma      

        self.results = None
        self.columns = None

    def load_data(self, data, columnDictionary):
        """
        Load and preprocess user-provided data.

        Parameters:
        - data: Pandas dataframe or path to csv file
        - columnDictionary: Dictionary mapping required parameters to column names.
        """
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise ValueError("Data must be a file path or pandas DataFrame")
        
        missing = [k for k, v in columnDictionary.items() if v not in self.data.columns]
        if missing:
            raise ValueError(f"Missing columns in the dataset: {missing}")
        
        self.columns = columnDictionary

        initial_row = self.data.iloc[0]
        self.N = initial_row[self.columns["population"]]
        self.I0 = initial_row[self.columns["new_infections"]]
        self.S0 = self.N - self.I0
        self.E0 = 0
        self.R0 = 0

        self.b = initial_row[self.columns["number_of_births"]]
        self.d = initial_row[self.columns["natural_death_rate"]] / 100000
        self.vs = initial_row[self.columns["viral_suppression"]]

    def simulate(self, years, initial_conditions=None, params=None, optimal_params=None, param_set=None):
        """
        Simulate the SEIRS model using the provided data and parameters.

        Parameters:
        - years: Number of years to simulate.
        - initial_conditions: Optional list of initial conditions [S, E, I, R].
        - params: Optional list of parameters (beta, sigma, nu, mu, delta, gamma) to use for simulation.
        - optimal_params: Optional dictionary of parameter sets.
        - param_set: The key of the parameter set to use from optimal_params.

        Returns:
        - results: Pandas DataFrame with simulation results.
        """
        if self.data is None:
            raise ValueError("Data must be loaded before simulation.")

        if optimal_params and param_set:
            params = optimal_params.get(param_set)
            if params is None:
                raise ValueError(f"Parameter set '{param_set}' not found in optimal_params.")
            self.beta, self.sigma, self.nu, self.mu, self.delta, self.gamma = params

        if params is not None:
            self.beta, self.sigma, self.nu, self.mu, self.delta, self.gamma = params

        if initial_conditions is None:
            initial_conditions = [self.S0, self.E0, self.I0, self.R0]
        
        def seirs(t, y):
            S, E, I, R = y
            N = S + E + I + R
            effective_nu = self.nu * self.vs

            dS_dt = self.b - self.beta * S * I / N - self.d * S + self.gamma * R
            dE_dt = self.beta * S * I / N - self.sigma * E - self.d * E
            dI_dt = self.sigma * E - effective_nu * I - self.d * I - self.delta * I
            dR_dt = effective_nu * I - self.gamma * R - self.d * R

            return [dS_dt, dE_dt, dI_dt, dR_dt]

        t_span = [0, years - 1]
        t_eval = np.linspace(0, years - 1, years)

        solution = solve_ivp(seirs, t_span, initial_conditions, t_eval=t_eval, method="RK45")

        S, E, I, R = solution.y
        N = S + E + I + R
        new_infections = self.beta * S * I / N
        hiv_deaths = self.delta * I

        self.results = pd.DataFrame({
            "Year": pd.Series(range(self.data[self.columns["year"]].iloc[-1] + 1, 
                                    self.data[self.columns["year"]].iloc[-1] + 1 + years)),
            "New HIV Infections": new_infections,
            "Exposed": E,
            "Infectious": I,
            "Recovered (ART)": R,
            "HIV Deaths": hiv_deaths
        })

        return self.results


    def plot_results(self):
        """
        Plot the simulation results.
        """
        if self.results is None:
            raise ValueError("No results to plot. Run simulate() first.")

        plt.figure(figsize=(12, 8))
        plt.plot(self.results["Year"], self.results["New HIV Infections"], label="New HIV Infections")
        plt.plot(self.results["Year"], self.results["Exposed"], label="Exposed")
        plt.plot(self.results["Year"], self.results["Infectious"], label="Infectious")
        plt.plot(self.results["Year"], self.results["Recovered (ART)"], label="Recovered (ART)")
        plt.plot(self.results["Year"], self.results["HIV Deaths"], label="HIV Deaths")
        plt.xlabel("Year")
        plt.ylabel("Population / Count")
        plt.title("SEIRS Model Simulation with Open Cohort Dynamics")
        plt.legend()
        plt.grid()
        plt.show()
