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
        # Initialize model parameters
        self.beta = beta
        self.sigma = sigma
        self.nu = nu
        self.mu = mu
        self.delta = delta
        self.gamma = gamma

        # Placeholder for simulation results, column mappings, and parameter samples
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
        # Check if the input data is a path or a DataFrame
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise ValueError("Data must be a file path or pandas DataFrame")

        # Ensure that all required columns exist in the data
        missing = [k for k, v in columnDictionary.items() if v not in self.data.columns]
        if missing:
            raise ValueError(f"Missing columns in the dataset: {missing}")

        # Map the column names from the dictionary
        self.columns = columnDictionary
        # Extract the first row of data for initializing model compartments and parameters
        initial_row = self.data.iloc[0]

        # Initialize population and compartment states
        self.N = initial_row[self.columns["population"]]  # Total US population
        self.I0 = initial_row[
            self.columns["new_infections"]
        ]  # Initial infectious population
        self.S0 = self.N - self.I0  # Initial susceptible population
        self.E0 = (
            self.S0 * 0.001
        )  # Initial exposed population, exposure rate is around 0.1%
        self.R0 = (
            initial_row[self.columns["total_hiv"]]
            * initial_row[self.columns["viral_suppression"]]
        )  # Recovered population (virally suppressed)
        self.D0 = initial_row[self.columns["deaths_hiv"]]  # Initial deaths from HIV

        # Initialize demographic and ART-related parameters
        self.b = initial_row[
            self.columns["number_of_births"]
        ]  # Annual number of births
        self.d = (
            initial_row[self.columns["natural_death_rate"]] / 100000
        )  # Death rate per individual
        self.art = initial_row[
            self.columns["viral_suppression"]
        ]  # Initial ART (viral suppression) proportion

    def simulate(self, years, initial_conditions=None, params=None):
        """
        Simulate the SEIRS model using the provided data and parameters.

        Parameters:
        - years: Number of years to simulate.
        - initial_conditions: Optional list of initial conditions [S, E, I, R].
        - params: Optional list of parameters (beta, sigma, nu, mu, delta, gamma) to use for simulation.

        Returns:
        - results: Pandas DataFrame with simulation results.
        """
        # Ensure that the data has been loaded before running the simulation
        if not hasattr(self, "data"):
            raise ValueError("Data must be loaded before simulation.")
        # If parameters are provided, update the model's parameters
        if params is not None:
            self.beta, self.sigma, self.nu, self.mu, self.delta, self.gamma = params

        # If initial conditions are not provided, use the default values from the loaded data
        if initial_conditions is None:
            initial_conditions = [self.S0, self.E0, self.I0, self.R0, self.D0]

        # Define the system of differential equations for the SEIRS model
        def seirs(t, y):
            S, E, I, R, D = (
                y  # Unpack compartments: Susceptible, Exposed, Infectious, Recovered, Deaths
            )
            N = S + E + I + R  # Total population (excluding deaths)

            # Compute effective ART recovery rate based on viral suppression proportion
            effective_nu = self.nu * self.art

            # Differential equations for the SEIRS model
            dS_dt = (
                self.b - self.beta * S * I / N - self.mu * S + self.gamma * R
            )  # Susceptible
            dE_dt = self.beta * S * I / N - self.sigma * E - self.mu * E  # Exposed
            dI_dt = (
                self.sigma * E - effective_nu * I - self.mu * I - self.delta * I
            )  # Infectious
            dR_dt = effective_nu * I - self.gamma * R - self.mu * R  # Recovered
            dD_dt = self.delta * I  # Deaths from HIV (cumulative)

            return [dS_dt, dE_dt, dI_dt, dR_dt, dD_dt]

        # Define the simulation time span and evaluation points
        t_span = [0, years - 1]
        t_eval = np.linspace(0, years - 1, years)

        # Solve the system of differential equations using the RK45 method
        solution = solve_ivp(
            seirs, t_span, initial_conditions, t_eval=t_eval, method="RK45"
        )

        # Compile the simulation results into a DataFrame
        self.results = pd.DataFrame(
            {
                "Year": pd.Series(
                    range(
                        self.data[self.columns["year"]].iloc[-1] + 1,
                        self.data[self.columns["year"]].iloc[-1] + 1 + years,
                    )
                ),
                "Exposed": solution.y[1],
                "Infectious": solution.y[2],
                "Recovered (ART)": solution.y[3],
                "Deaths (HIV)": solution.y[4],
            }
        )

        return self.results

    def calibrate_parameters(self, num_bootstrap=100):
        """
        Calibrate model parameters using bootstrapped historical data.

        Returns:
        - Mean simulated values for infections, recovered, and deaths for the historical years.
        - Final MSE for infections and deaths.
        """
        # Store results for all bootstrap samples
        all_simulated_infections = []
        all_simulated_recovered = []
        all_simulated_deaths = []

        def loss_function(params, data):
            # Simulate the SEIRS model for the given parameters
            simulated = self.simulate(len(data), params=params)

            # Extract observed values from the data
            observed_infections = data[self.columns["new_infections"]].values
            observed_deaths = data[self.columns["deaths_hiv"]].values
            observed_virally_suppressed = (
                data[self.columns["viral_suppression"]]
                * data[self.columns["total_hiv"]]
            ).values

            # Store simulated results for analysis after bootstrapping
            all_simulated_infections.append(simulated["Infectious"].values)
            all_simulated_recovered.append(simulated["Recovered (ART)"].values)
            all_simulated_deaths.append(
                simulated["Deaths (HIV)"].diff().fillna(0).values
            )

            # Calculate MSE for infections, recovered, and deaths
            mse_infections = (
                mean_squared_error(observed_infections, simulated["Infectious"].values)
                / observed_infections.mean()
            )
            mse_recovered = (
                mean_squared_error(
                    observed_virally_suppressed, simulated["Recovered (ART)"].values
                )
                / observed_virally_suppressed.mean()
            )
            mse_deaths = (
                mean_squared_error(
                    observed_deaths, simulated["Deaths (HIV)"].diff().fillna(0).values
                )
                / observed_deaths.mean()
            )

            # Return the total loss as the sum of the MSEs
            return mse_infections + mse_deaths + mse_recovered

        # Initial guesses for parameters and their bounds for optimization
        initial_params = [
            self.beta,
            self.sigma,
            self.nu,
            self.mu,
            self.delta,
            self.gamma,
        ]
        bounds = [(0, 0.1), (0, 0.1), (0, 1), (0, 1), (0, 1), (0, 0.1)]

        # Perform bootstrap calibration by sampling data and minimizing the loss function
        self.param_samples = []
        for _ in range(num_bootstrap):
            bootstrap_sample = self.data.sample(frac=1, replace=True)
            result = minimize(
                loss_function,
                initial_params,
                args=(bootstrap_sample,),
                bounds=bounds,
                method="L-BFGS-B",
            )
            if result.success:
                self.param_samples.append(result.x)

        # Compute the mean of the optimized parameters from all bootstrap samples
        self.param_samples = np.array(self.param_samples)
        self.beta, self.sigma, self.nu, self.mu, self.delta, self.gamma = np.mean(
            self.param_samples, axis=0
        )
        print("Calibration completed with bootstrapping.")
        print(
            "Mean calibrated parameters:",
            self.beta,
            self.sigma,
            self.nu,
            self.mu,
            self.delta,
            self.gamma,
        )

        # Calculate mean simulated values for infections, recovered, and deaths
        mean_simulated_infections = np.mean(all_simulated_infections, axis=0)
        mean_simulated_recovered = np.mean(all_simulated_recovered, axis=0)
        mean_simulated_deaths = np.mean(all_simulated_deaths, axis=0)

        # Extract observed values for comparison with simulated results
        observed_infections = self.data[self.columns["new_infections"]].values
        observed_virally_suppressed = (
            self.data[self.columns["viral_suppression"]]
            * self.data[self.columns["total_hiv"]]
        ).values
        observed_deaths = self.data[self.columns["deaths_hiv"]].values

        # Calculate final MSE for each metric
        final_mse_infections = mean_squared_error(
            observed_infections, mean_simulated_infections
        )
        final_mse_deaths = mean_squared_error(observed_deaths, mean_simulated_deaths)
        final_mse_recovered = mean_squared_error(
            observed_virally_suppressed, mean_simulated_recovered
        )

        # Print final MSE values for evaluation
        print(f"Final MSE (Infections): {final_mse_infections:.2f}")
        print(f"Final MSE (Death): {final_mse_deaths:.2f}")
        print(f"Final MSE (Recovered): {final_mse_recovered:.2f}")

        # Plot observed vs. mean simulated new infections
        years = self.data["Year"].values

        # Plot for new infections
        plt.figure(figsize=(12, 6))
        plt.plot(
            years,
            observed_infections / 1000,
            label="Observed New Infections",
            marker="o",
        )
        plt.plot(
            years,
            mean_simulated_infections / 1000,
            label="Mean Simulated New Infections",
            linestyle="--",
        )
        plt.xlabel("Year")
        plt.ylabel("New Infections (thousands)")
        plt.title("Observed vs. Mean Simulated New Infections")
        plt.legend()
        plt.grid()
        plt.ticklabel_format(style="plain")
        plt.savefig("infections_fit3.png")
        plt.show()

        # Plot for deaths
        plt.figure(figsize=(12, 6))
        plt.plot(years, observed_deaths / 1000, label="Observed Deaths", marker="o")
        plt.plot(
            years,
            mean_simulated_deaths / 1000,
            label="Mean Simulated Deaths",
            linestyle="--",
        )
        plt.xlabel("Year")
        plt.ylabel("HIV-Related Deaths (thousands)")
        plt.title("Observed vs. Mean Simulated HIV-Related Deaths")
        plt.legend()
        plt.grid()
        plt.ticklabel_format(style="plain")
        plt.savefig("deaths_fit3.png")
        plt.show()

        # Plot for recovered (virally suppressed)
        plt.figure(figsize=(12, 6))
        plt.plot(
            years,
            observed_virally_suppressed / 1000,
            label="Observed Recovered",
            marker="o",
        )
        plt.plot(
            years,
            mean_simulated_recovered / 1000,
            label="Mean Simulated Recovered",
            linestyle="--",
        )
        plt.xlabel("Year")
        plt.ylabel("Number of Recovered (thousands)")
        plt.title("Observed vs. Mean Simulated Recovered")
        plt.legend()
        plt.grid()
        plt.ticklabel_format(style="plain")
        plt.savefig("recovered_fit3.png")
        plt.show()

    def simulate_with_uncertainty(self, years):
        # Check if parameter samples exist; calibration must be run first
        if not hasattr(self.param_samples, "size"):
            raise ValueError(
                "No parameter samples found. Run calibrate_parameters() first."
            )
        if self.param_samples.size == 0:
            raise ValueError(
                "No parameter samples found. Run calibrate_parameters() first."
            )

        # Extract the last row of the data for initializing the simulation
        last_row = self.data.iloc[-1]

        # Initialize population and SEIRS compartments based on the last observed data
        N = last_row[self.columns["population"]]  # Total population
        I0 = last_row[self.columns["new_infections"]]  # Initial infectious individuals
        S0 = N - I0  # Initial susceptible population
        E0 = S0 * 0.000667  # Initial exposed population
        R0 = (
            last_row[self.columns["total_hiv"]]
            * last_row[self.columns["viral_suppression"]]
        )  # Recovered (virally suppressed)
        D0 = last_row[self.columns["deaths_hiv"]]  # Initial cumulative deaths

        # List to store simulation results for each parameter sample
        all_results = []
        # Define the initial conditions for the simulation
        init_conds = [S0, E0, I0, R0, D0]
        # Run simulations for each parameter sample
        for params in self.param_samples:
            result = self.simulate(years, initial_conditions=init_conds, params=params)
            all_results.append(
                result[
                    ["Exposed", "Infectious", "Recovered (ART)", "Deaths (HIV)"]
                ].values
            )

        # Convert all simulation results into a single numpy array
        all_results = np.array(all_results)
        # Calculate mean and confidence intervals (2.5% and 97.5%) across all simulations
        mean_results = np.mean(all_results, axis=0)
        lower_ci = np.percentile(all_results, 2.5, axis=0)
        upper_ci = np.percentile(all_results, 97.5, axis=0)

        years = result["Year"]

        # Define compartment names and corresponding colors for plotting
        compartments = ["Exposed", "Infectious", "Recovered (ART)", "Deaths (HIV)"]
        colors = ["orange", "red", "green", "purple"]

        # Plot the mean and confidence intervals for each compartment
        plt.figure(figsize=(12, 8))
        for i, compartment in enumerate(compartments):
            plt.plot(
                years,
                mean_results[:, i] / 1000,
                label=f"Mean {compartment}",
                color=colors[i],
            )
            plt.fill_between(
                years,
                lower_ci[:, i] / 1000,
                upper_ci[:, i] / 1000,
                color=colors[i],
                alpha=0.2,
            )

        # Add labels, title, and formatting to the plot
        plt.xlabel("Year")
        plt.ylabel("Population (thousands)")
        plt.title("HIV SEIRS Model (95% Confidence Intervals)")
        plt.ticklabel_format(style="plain")
        plt.legend()
        plt.grid()
        plt.savefig("SEIRS_results3.png")
        plt.show()
