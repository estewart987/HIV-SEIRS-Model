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
        self.param_samples = []

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
        self.D0 = 0 

        self.b = initial_row[self.columns["number_of_births"]]
        self.d = initial_row[self.columns["natural_death_rate"]] / 100000
        self.art = initial_row[self.columns["viral_suppression"]]

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

        if params is not None:
            self.beta, self.sigma, self.nu, self.mu, self.delta, self.gamma = params

        if initial_conditions is None:
            initial_conditions = [self.S0, self.E0, self.I0, self.R0, self.D0]
    
        def seirs(t, y):
            S, E, I, R, D = y
            N = S + E + I + R

            effective_nu = self.nu * self.art

            dS_dt = self.b - self.beta * S * I / N - self.mu * S + self.gamma * R
            dE_dt = self.beta * S * I / N - self.sigma * E - self.mu * E
            dI_dt = self.sigma * E - effective_nu * I - self.mu * I - self.delta * I
            dR_dt = effective_nu * I - self.gamma * R - self.mu * R
            dD_dt = self.delta * I 

            return [dS_dt, dE_dt, dI_dt, dR_dt, dD_dt]

        t_span = [0, years - 1]
        t_eval = np.linspace(0, years - 1, years)

        solution = solve_ivp(seirs, t_span, initial_conditions, t_eval=t_eval, method="RK45")

        self.results = pd.DataFrame({
            "Year": pd.Series(range(self.data[self.columns["year"]].iloc[-1] + 1, 
                                    self.data[self.columns["year"]].iloc[-1] + 1 + years)),
            "Exposed": solution.y[1],
            "Infectious": solution.y[2],
            "Recovered (ART)": solution.y[3],
            "Deaths (HIV)": solution.y[4]
        })

        return self.results

    def calibrate_parameters(self, num_bootstrap=100):
        """
        Calibrate model parameters using bootstrapped historical data.
        """
        def loss_function(params, data):
            simulated = self.simulate(len(data), params=params)
            observed_infections = data[self.columns["new_infections"]].values
            simulated_infections = simulated["Infectious"].values
            return mean_squared_error(observed_infections, simulated_infections)

        initial_params = [self.beta, self.sigma, self.nu, self.mu, self.delta, self.gamma]
        bounds = [(0, 1), (0, 1), (0, 1), (0, 0.1), (0, 0.1), (0, 1)]

        self.param_samples = []

        for _ in range(num_bootstrap):
            bootstrap_sample = self.data.sample(frac=1, replace=True)
            result = minimize(loss_function, initial_params, args=(bootstrap_sample,), bounds=bounds, method="L-BFGS-B")
            if result.success:
                self.param_samples.append(result.x)

        self.param_samples = np.array(self.param_samples)
        self.beta, self.sigma, self.nu, self.mu, self.delta, self.gamma = np.mean(self.param_samples, axis=0)
        print("Calibration completed with bootstrapping.")
        print("Mean calibrated parameters:", self.beta, self.sigma, self.nu, self.mu, self.delta, self.gamma)

    def simulate_with_uncertainty(self, years):
        if self.param_samples.size == 0:
            raise ValueError("No parameter samples found. Run calibrate_parameters() first.")

        all_results = []

        for params in self.param_samples:
            result = self.simulate(years, params=params)
            all_results.append(result[["Exposed", "Infectious", "Recovered (ART)", "Deaths (HIV)"]].values)

        all_results = np.array(all_results)
        mean_results = np.mean(all_results, axis=0)
        lower_ci = np.percentile(all_results, 2.5, axis=0)
        upper_ci = np.percentile(all_results, 97.5, axis=0)

        years = result["Year"]

        compartments = ["Exposed", "Infectious", "Recovered (ART)", "Deaths (HIV)"]
        colors = ["orange", "red", "green", "purple"]

        plt.figure(figsize=(12, 8))
        for i, compartment in enumerate(compartments):
            plt.plot(years, mean_results[:, i], label=f"Mean {compartment}", color=colors[i])
            plt.fill_between(years, lower_ci[:, i], upper_ci[:, i], color=colors[i], alpha=0.2)

        plt.xlabel("Year")
        plt.ylabel("Population")
        plt.title("SEIRS Model Simulation with 95% Confidence Intervals")
        plt.legend()
        plt.grid()
        plt.show()
